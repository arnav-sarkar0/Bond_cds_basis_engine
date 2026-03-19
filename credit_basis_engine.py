"""
Bond–CDS Consistency & Basis Engine
=====================================
Phase 1 : Hazard rate bootstrapping from bond prices
Phase 2 : CDS pricing using same h(t)  — matches C++ engine exactly
Phase 3 : Basis detection & trade signal table

Data sources (priority order):
  1. CLI flags:   --bonds bonds.csv --cds cds_market.csv
  2. Config block below (BONDS / MARKET_CDS dicts)

Math (mirrors C++ engine exactly)
----------------------------------
Bond discount  : exp(-(r*t + L*H(t)))   L = 1-R
CDS survival   : exp(-H(t))             (no L)
CDS fair spread: protectionLeg / premiumLeg
"""

import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize


# ══════════════════════════════════════════════════════════════
# CONFIG  — used when no CSV is supplied
# ══════════════════════════════════════════════════════════════

RISK_FREE   = 0.05
RECOVERY    = 0.40
FACE        = 100.0
COUPON      = 5.0
CDS_STEPS   = 1000
SMOOTH      = True
SMOOTH_PEN  = 10.0

DEFAULT_BONDS = [
    {"maturity": 1, "coupon": COUPON, "face": FACE, "price": 95.0},
    {"maturity": 2, "coupon": COUPON, "face": FACE, "price": 93.0},
    {"maturity": 3, "coupon": COUPON, "face": FACE, "price": 90.0},
    {"maturity": 4, "coupon": COUPON, "face": FACE, "price": 85.0},
    {"maturity": 5, "coupon": COUPON, "face": FACE, "price": 80.0},
]

DEFAULT_MARKET_CDS = {
    1: 0.0110,
    2: 0.0140,
    3: 0.0170,
    4: 0.0200,
    5: 0.0220,
}


# ══════════════════════════════════════════════════════════════
# CSV LOADERS
# ══════════════════════════════════════════════════════════════

def load_bonds_csv(path: str) -> list[dict]:
    """
    Expected columns: maturity, coupon, face, market_price
    coupon and face are optional — fall back to CONFIG values.
    """
    bonds = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bonds.append({
                "maturity": int(row["maturity"]),
                "coupon":   float(row.get("coupon",  COUPON)),
                "face":     float(row.get("face",    FACE)),
                "price":    float(row["market_price"]),
            })
    bonds.sort(key=lambda b: b["maturity"])
    return bonds


def load_cds_csv(path: str) -> dict[int, float]:
    """
    Expected columns: maturity, spread_bps
    Returns {maturity: spread_decimal}
    """
    cds = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cds[int(row["maturity"])] = float(row["spread_bps"]) / 10_000.0
    return cds


# ══════════════════════════════════════════════════════════════
# C++ ENGINE CALL  (optional)
# ══════════════════════════════════════════════════════════════

def call_engine(hazards: np.ndarray, maturity: int) -> tuple[float, float]:
    import subprocess
    hazards = np.asarray(hazards, dtype=float)
    if np.any(~np.isfinite(hazards)) or np.any(hazards < 0):
        return 1e10, 1e10
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    engine_path = os.path.join(base_dir, "engine.exe")
    result = subprocess.run(
        [engine_path, *map(str, hazards), str(maturity)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return 1e10, 1e10
    tokens = result.stdout.strip().split()
    if len(tokens) < 2:
        return 1e10, 1e10
    return float(tokens[0]), float(tokens[1])


# ══════════════════════════════════════════════════════════════
# PURE-PYTHON PRICERS  (mirror C++ math exactly)
# ══════════════════════════════════════════════════════════════

def _hazard_integral(hazards: np.ndarray, t: float) -> float:
    """H(t) = ∫₀ᵗ h(s) ds — mirrors HazardCurve::integral()"""
    result, prev = 0.0, 0.0
    for i, t_i in enumerate(range(1, len(hazards) + 1)):
        dt = min(t, float(t_i)) - prev
        if dt > 0:
            result += hazards[i] * dt
        prev = float(t_i)
        if t_i >= t:
            break
    return result


def py_bond_price(
    hazards:  np.ndarray,
    coupon:   float = None,
    face:     float = None,
    r:        float = RISK_FREE,
    recovery: float = RECOVERY,
) -> float:
    """Mirrors BondPricer::price() — discount(t) = exp(-(r*t + L*H(t)))"""
    coupon = coupon if coupon is not None else COUPON
    face   = face   if face   is not None else FACE
    L      = 1.0 - recovery
    T      = len(hazards)
    price  = 0.0
    for i in range(T):
        t   = float(i + 1)
        H_t = _hazard_integral(hazards, t)
        d   = np.exp(-(r * t + L * H_t))
        cf  = coupon + (face if i == T - 1 else 0.0)
        price += cf * d
    return price


def py_cds_spread(
    hazards:  np.ndarray,
    maturity: float,
    r:        float = RISK_FREE,
    recovery: float = RECOVERY,
    steps:    int   = CDS_STEPS,
) -> float:
    """Mirrors CDSPricer::fairSpread() exactly."""
    L            = 1.0 - recovery
    dt           = maturity / steps
    premium_leg  = 0.0
    protect_leg  = 0.0
    for i in range(1, steps + 1):
        t        = i * dt
        H_t      = _hazard_integral(hazards, t)
        discount = np.exp(-r * t)
        survival = np.exp(-H_t)
        idx      = min(int(np.ceil(t)) - 1, len(hazards) - 1)
        h_t      = hazards[idx]
        premium_leg += discount * survival * dt
        protect_leg += discount * h_t * survival * L * dt
    return 0.0 if premium_leg == 0 else protect_leg / premium_leg


# ══════════════════════════════════════════════════════════════
# PHASE 1 — BOOTSTRAP
# ══════════════════════════════════════════════════════════════

def bootstrap_hazard(bonds: list[dict]) -> np.ndarray:
    print("\n[Phase 1] Bootstrapping hazard rates")
    print(f"  {'Mat':>4}  {'Coupon':>8}  {'h_i':>10}  {'Model P':>10}  {'Market P':>10}")
    print("  " + "-" * 52)
    hazards = []
    for bond in bonds:
        m      = bond["maturity"]
        mp     = bond["price"]
        coupon = bond.get("coupon", COUPON)
        face   = bond.get("face",   FACE)

        def residual(h):
            return py_bond_price(np.array(hazards + [h]), coupon=coupon, face=face) - mp

        h_i = brentq(residual, 1e-6, 0.9999, xtol=1e-10, maxiter=300)
        hazards.append(h_i)
        check = py_bond_price(np.array(hazards), coupon=coupon, face=face)
        print(f"  {m:>4}Y  {coupon:>8.4f}  {h_i:>10.6f}  {check:>10.4f}  {mp:>10.4f}")
    return np.array(hazards)


# ══════════════════════════════════════════════════════════════
# PHASE 1b — SMOOTH
# ══════════════════════════════════════════════════════════════

def smooth_hazards(hazard_init: np.ndarray, bonds: list[dict], penalty: float = SMOOTH_PEN) -> np.ndarray:
    def objective(h):
        h   = np.asarray(h)
        err = sum(
            (py_bond_price(h[:b["maturity"]], coupon=b.get("coupon", COUPON), face=b.get("face", FACE)) - b["price"]) ** 2
            for b in bonds
        )
        smooth = sum((h[i+1] - h[i])**2 for i in range(len(h)-1))
        return err + penalty * smooth

    res = minimize(objective, hazard_init, method="L-BFGS-B",
                   bounds=[(1e-6, 0.9999)] * len(hazard_init),
                   options={"maxiter": 1000, "ftol": 1e-14})
    if not res.success:
        print(f"  [Warning] Smoother: {res.message}")
    return res.x


# ══════════════════════════════════════════════════════════════
# PHASE 2 — SURVIVAL & DEFAULT
# ══════════════════════════════════════════════════════════════

def survival_and_default(hazards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H  = np.array([_hazard_integral(hazards, float(t)) for t in range(1, len(hazards)+1)])
    S  = np.exp(-H)
    return S, 1.0 - S


# ══════════════════════════════════════════════════════════════
# PHASE 3 — BASIS ENGINE
# ══════════════════════════════════════════════════════════════

def compute_basis(hazards: np.ndarray, bonds: list[dict], market_cds: dict) -> list[dict]:
    results = []
    for bond in bonds:
        m       = bond["maturity"]
        mkt_cds = market_cds.get(m)
        if mkt_cds is None:
            continue
        model_cds = py_cds_spread(hazards[:m], float(m))
        basis_bps = (mkt_cds - model_cds) * 10_000
        if basis_bps > 5:
            signal = "Buy Bond / Sell CDS protection"
        elif basis_bps < -5:
            signal = "Sell Bond / Buy CDS protection"
        else:
            signal = "Fairly Priced — no trade"
        results.append({
            "maturity":   m,
            "model_cds":  model_cds,
            "market_cds": mkt_cds,
            "basis_bps":  basis_bps,
            "signal":     signal,
        })
    return results


def print_basis_table(results: list[dict]) -> None:
    print("\n── Bond–CDS Basis Table ──────────────────────────────────────────────────")
    print(f"  {'Mat':>4}  {'Model CDS':>10}  {'Market CDS':>10}  {'Basis':>10}  Signal")
    print("  " + "-" * 72)
    for r in results:
        print(
            f"  {r['maturity']:>4}Y"
            f"  {r['model_cds']*10000:>9.1f}bp"
            f"  {r['market_cds']*10000:>9.1f}bp"
            f"  {r['basis_bps']:>+9.1f}bp"
            f"  {r['signal']}"
        )


def check_bond_arbitrage(hazards: np.ndarray, bonds: list[dict]) -> None:
    print("\n── Bond Arbitrage Check ──────────────────────────────────────────────────")
    print(f"  {'Mat':>4}  {'Market':>8}  {'Model':>8}  {'Diff':>8}  Signal")
    print("  " + "-" * 52)
    for bond in bonds:
        m       = bond["maturity"]
        P_mkt   = bond["price"]
        P_model = py_bond_price(hazards[:m], coupon=bond.get("coupon", COUPON), face=bond.get("face", FACE))
        diff    = P_mkt - P_model
        sig     = ("Overpriced  → Sell/Short" if diff >  0.10 else
                   "Underpriced → Buy"         if diff < -0.10 else
                   "Fairly Priced")
        print(f"  {m:>4}Y  {P_mkt:>8.2f}  {P_model:>8.2f}  {diff:>+8.3f}  {sig}")


# ══════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════

def plot_all(mats, hazard_raw, hazard_final, S, PD, basis_results, bonds, title_tag=""):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    tag = f"  [{title_tag}]" if title_tag else ""
    fig.suptitle(f"Bond–CDS Basis Engine  |  Duffie–Singleton{tag}", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(mats, hazard_raw, "o--", color="#378ADD", label="Bootstrapped", linewidth=1.5)
    if not np.allclose(hazard_raw, hazard_final):
        ax.plot(mats, hazard_final, "s-", color="#1D9E75", label="Smoothed", linewidth=2)
    ax.set_title("Hazard Rate Curve  h(t)")
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Hazard Rate")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(mats, S,  "o-", color="#1D9E75", label="Survival  S(t)", linewidth=2)
    ax.plot(mats, PD, "s-", color="#D85A30", label="Cum. PD",        linewidth=2)
    ax.set_title("Survival & Default Probability")
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Probability")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    b_mats = [r["maturity"] for r in basis_results]
    ax.plot(b_mats, [r["model_cds"]  * 10000 for r in basis_results], "o-",  color="#1D9E75", label="Model CDS",  linewidth=2)
    ax.plot(b_mats, [r["market_cds"] * 10000 for r in basis_results], "s--", color="#D85A30", label="Market CDS", linewidth=2)
    ax.set_title("CDS Spread: Model vs Market (bps)")
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Spread (bps)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    basis_vals = [r["basis_bps"] for r in basis_results]
    ax.bar(b_mats, basis_vals, color=["#D85A30" if b > 0 else "#378ADD" for b in basis_vals], width=0.5, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Basis  (Market − Model CDS)  bps")
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Basis (bps)")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    mkt_prices   = [b["price"] for b in bonds]
    model_prices = [py_bond_price(hazard_final[:b["maturity"]], coupon=b.get("coupon", COUPON), face=b.get("face", FACE)) for b in bonds]
    ax.plot(mats, mkt_prices,   "o-",  color="#378ADD", label="Market Price", linewidth=2)
    ax.plot(mats, model_prices, "s--", color="#D85A30", label="Model Price",  linewidth=2)
    ax.set_title("Bond Price: Market vs Model")
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Price")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.bar(mats, hazard_final, width=0.5, color="#5DCAA5", edgecolor="white")
    ax.set_title("Incremental Hazard Rates")
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("h(t)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "basis_engine_dashboard.png")
    plt.savefig(out, dpi=150)
    print(f"\n  Chart saved → {out}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Bond–CDS Basis Engine")
    p.add_argument("--bonds",     default=None, help="Path to bonds.csv")
    p.add_argument("--cds",       default=None, help="Path to cds_market.csv")
    p.add_argument("--no-smooth", action="store_true", help="Skip smoothing")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.bonds:
        print(f"  Loading bonds from : {args.bonds}")
        bonds = load_bonds_csv(args.bonds)
    else:
        print("  Using default bond config")
        bonds = DEFAULT_BONDS

    if args.cds:
        print(f"  Loading CDS from   : {args.cds}")
        market_cds = load_cds_csv(args.cds)
    else:
        print("  Using default CDS config")
        market_cds = DEFAULT_MARKET_CDS

    do_smooth = SMOOTH and not args.no_smooth
    mats      = np.array([b["maturity"] for b in bonds], dtype=float)
    title_tag = os.path.basename(args.bonds).replace(".csv","") if args.bonds else "default"

    # Phase 1
    hazard_raw = bootstrap_hazard(bonds)

    # Phase 1b
    if do_smooth:
        print("\n[Phase 1b] Smoothing hazard rates …")
        hazard_final = smooth_hazards(hazard_raw, bonds)
        print(f"  Raw      : {np.round(hazard_raw,   6)}")
        print(f"  Smoothed : {np.round(hazard_final, 6)}")
    else:
        hazard_final = hazard_raw

    # Phase 2
    print("\n[Phase 2] Survival & Default Probabilities")
    S, PD = survival_and_default(hazard_final)
    print(f"  {'Mat':>4}  {'S(t)':>8}  {'PD(t)':>8}")
    for i, m in enumerate(mats):
        print(f"  {int(m):>4}Y  {S[i]:>8.4f}  {PD[i]:>8.4f}")

    print("\n[Phase 2] Model CDS Spreads")
    for bond in bonds:
        m   = bond["maturity"]
        cds = py_cds_spread(hazard_final[:m], float(m))
        print(f"  {m}Y  model CDS = {cds*10000:.2f} bps")

    # Phase 3
    print("\n[Phase 3] Basis Engine")
    basis_results = compute_basis(hazard_final, bonds, market_cds)
    print_basis_table(basis_results)
    check_bond_arbitrage(hazard_final, bonds)

    # Plots
    print("\n[Plots] Building dashboard …")
    plot_all(mats, hazard_raw, hazard_final, S, PD, basis_results, bonds, title_tag)


if __name__ == "__main__":
    main()
