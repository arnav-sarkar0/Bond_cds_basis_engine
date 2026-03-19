"""
Market Data Generator
======================
Generates realistic corporate bond prices and CDS spreads based on:
  - Credit rating  (AAA → CCC)
  - Sector         (financials, energy, tech, industrials, utilities)
  - Scenario       (normal, stress, crisis)

Outputs:
  bonds.csv      — maturity, coupon, face, market_price
  cds_market.csv — maturity, spread_bps

Usage:
  python generate_market_data.py
  python generate_market_data.py --rating BB --sector energy --scenario stress
"""

import argparse
import csv
import os
import numpy as np

# ══════════════════════════════════════════════════════════════
# PARAMETER TABLES
# ══════════════════════════════════════════════════════════════

# Base hazard rates (annualised) per rating — calibrated to
# historical Moody's default studies (approximate)
RATING_HAZARD = {
    "AAA": 0.0002,
    "AA":  0.0005,
    "A":   0.0010,
    "BBB": 0.0035,
    "BB":  0.0120,
    "B":   0.0300,
    "CCC": 0.0800,
}

# Hazard term-structure shape per rating
# (multipliers applied to each maturity bucket 1Y–5Y)
# Higher-rated names have flatter curves; lower-rated names steepen fast
RATING_TERM_SHAPE = {
    "AAA": [1.00, 1.05, 1.10, 1.12, 1.13],
    "AA":  [1.00, 1.08, 1.15, 1.20, 1.22],
    "A":   [1.00, 1.10, 1.20, 1.28, 1.33],
    "BBB": [1.00, 1.15, 1.30, 1.42, 1.50],
    "BB":  [1.00, 1.20, 1.40, 1.55, 1.65],
    "B":   [1.00, 1.25, 1.50, 1.70, 1.85],
    "CCC": [1.00, 1.30, 1.60, 1.85, 2.05],
}

# Sector spread multipliers on top of base hazard
SECTOR_MULTIPLIER = {
    "financials":  1.20,   # systemic risk premium
    "energy":      1.35,   # commodity + capex risk
    "tech":        0.85,   # typically tighter
    "industrials": 1.00,   # baseline
    "utilities":   0.75,   # regulated, defensive
}

# Scenario multipliers — applied to ALL hazard rates
SCENARIO_MULTIPLIER = {
    "normal": 1.00,
    "stress": 2.50,   # ~2008 levels for IG, worse for HY
    "crisis": 5.00,   # distressed / near-default environment
}

# CDS market basis adjustment per scenario (bps added to model spread)
# Reflects liquidity premium, supply/demand imbalances in CDS market
SCENARIO_CDS_BASIS = {
    "normal":  {"mean":  0.0, "std":  3.0},
    "stress":  {"mean": 15.0, "std": 10.0},
    "crisis":  {"mean": 40.0, "std": 25.0},
}

# Coupon = risk-free rate + credit spread (rough rule of thumb)
RISK_FREE_RATE = 0.05
FACE_VALUE     = 100.0
MATURITIES     = [1, 2, 3, 4, 5]
RECOVERY_RATE  = 0.40


# ══════════════════════════════════════════════════════════════
# CORE MATH  (self-contained, no dependency on main engine)
# ══════════════════════════════════════════════════════════════

def build_hazard_curve(rating: str, sector: str, scenario: str) -> np.ndarray:
    """
    Construct a piecewise-constant hazard curve for maturities 1–5Y.
    hazard[i] applies over (i, i+1].
    """
    base      = RATING_HAZARD[rating]
    sector_m  = SECTOR_MULTIPLIER[sector]
    scenario_m= SCENARIO_MULTIPLIER[scenario]
    shape     = RATING_TERM_SHAPE[rating]

    hazards = np.array([
        base * sector_m * scenario_m * shape[i]
        for i in range(len(MATURITIES))
    ])

    # Hard cap — hazard rate can't exceed ~95% per year
    return np.clip(hazards, 1e-6, 0.95)


def hazard_integral(hazards: np.ndarray, t: float) -> float:
    """H(t) = ∫₀ᵗ h(s) ds  for piecewise-constant h on integer intervals."""
    result, prev = 0.0, 0.0
    for i, t_i in enumerate(range(1, len(hazards) + 1)):
        dt = min(t, float(t_i)) - prev
        if dt > 0:
            result += hazards[i] * dt
        prev = float(t_i)
        if t_i >= t:
            break
    return result


def bond_price(
    hazards:  np.ndarray,
    maturity: int,
    coupon:   float,
    face:     float = FACE_VALUE,
    r:        float = RISK_FREE_RATE,
    recovery: float = RECOVERY_RATE,
) -> float:
    """
    Duffie–Singleton bond price.
    discount(t) = exp(-(r*t + L*H(t)))
    """
    L     = 1.0 - recovery
    price = 0.0
    for i in range(maturity):
        t   = float(i + 1)
        H_t = hazard_integral(hazards, t)
        d   = np.exp(-(r * t + L * H_t))
        cf  = coupon + (face if i == maturity - 1 else 0.0)
        price += cf * d
    return price


def cds_spread(
    hazards:  np.ndarray,
    maturity: int,
    r:        float = RISK_FREE_RATE,
    recovery: float = RECOVERY_RATE,
    steps:    int   = 1000,
) -> float:
    """Fair CDS spread — mirrors CDSPricer::fairSpread() exactly."""
    L            = 1.0 - recovery
    dt           = maturity / steps
    premium_leg  = 0.0
    protect_leg  = 0.0

    for i in range(1, steps + 1):
        t        = i * dt
        H_t      = hazard_integral(hazards, t)
        discount = np.exp(-r * t)
        survival = np.exp(-H_t)
        idx      = min(int(np.ceil(t)) - 1, len(hazards) - 1)
        h_t      = hazards[idx]

        premium_leg += discount * survival * dt
        protect_leg += discount * h_t * survival * L * dt

    return 0.0 if premium_leg == 0 else protect_leg / premium_leg


def implied_coupon(
    hazards:  np.ndarray,
    maturity: int,
    r:        float = RISK_FREE_RATE,
) -> float:
    """
    Coupon that prices the bond at par (face = 100).
    Solved analytically: C = (Face - Face*d(T)) / sum(d(t))
    where d(t) = exp(-(r*t + L*H(t)))
    """
    L       = 1.0 - RECOVERY_RATE
    discounts = []
    for i in range(maturity):
        t   = float(i + 1)
        H_t = hazard_integral(hazards, t)
        discounts.append(np.exp(-(r * t + L * H_t)))

    d_T    = discounts[-1]
    sum_d  = sum(discounts)
    coupon = FACE_VALUE * (1.0 - d_T) / sum_d
    return coupon


# ══════════════════════════════════════════════════════════════
# GENERATOR
# ══════════════════════════════════════════════════════════════

def generate(
    rating:   str,
    sector:   str,
    scenario: str,
    seed:     int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Returns:
      bonds_data : list of dicts for bonds.csv
      cds_data   : list of dicts for cds_market.csv
    """
    rng     = np.random.default_rng(seed)
    hazards = build_hazard_curve(rating, sector, scenario)

    bonds_data = []
    cds_data   = []

    basis_cfg = SCENARIO_CDS_BASIS[scenario]

    for m in MATURITIES:

        # ── Bond ────────────────────────────────────────────
        coupon      = implied_coupon(hazards[:m], m)
        clean_price = bond_price(hazards[:m], m, coupon)

        # Add small bid-ask / market noise (tighter for IG, wider for HY)
        noise_std   = 0.05 if rating in ("AAA","AA","A") else \
                      0.15 if rating in ("BBB","BB")     else 0.30
        market_price = clean_price + rng.normal(0, noise_std)
        market_price = max(market_price, 1.0)   # sanity floor

        bonds_data.append({
            "maturity":     m,
            "coupon":       round(coupon, 4),
            "face":         FACE_VALUE,
            "market_price": round(market_price, 4),
        })

        # ── CDS ─────────────────────────────────────────────
        model_s     = cds_spread(hazards[:m], m)
        basis_noise = rng.normal(basis_cfg["mean"], basis_cfg["std"]) / 10_000
        mkt_spread  = max(model_s + basis_noise, 1e-5)

        cds_data.append({
            "maturity":   m,
            "spread_bps": round(mkt_spread * 10_000, 2),
        })

    return bonds_data, cds_data


def save_csvs(
    bonds_data: list[dict],
    cds_data:   list[dict],
    out_dir:    str = ".",
) -> tuple[str, str]:

    bonds_path = os.path.join(out_dir, "bonds.csv")
    cds_path   = os.path.join(out_dir, "cds_market.csv")

    with open(bonds_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["maturity","coupon","face","market_price"])
        w.writeheader()
        w.writerows(bonds_data)

    with open(cds_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["maturity","spread_bps"])
        w.writeheader()
        w.writerows(cds_data)

    return bonds_path, cds_path


def print_summary(
    rating:     str,
    sector:     str,
    scenario:   str,
    bonds_data: list[dict],
    cds_data:   list[dict],
) -> None:

    hazards = build_hazard_curve(rating, sector, scenario)

    print(f"\n{'═'*62}")
    print(f"  Issuer Profile: {rating} rated  |  {sector.title()}  |  {scenario.upper()} scenario")
    print(f"{'═'*62}")

    print(f"\n  Hazard Rates (annualised)")
    print(f"  {'Mat':>4}  {'h(t)':>8}  {'H(t)':>8}  {'S(t)':>8}  {'PD(t)':>8}")
    print("  " + "-" * 46)
    for i, m in enumerate(MATURITIES):
        H_t = hazard_integral(hazards, float(m))
        S_t = np.exp(-H_t)
        print(f"  {m:>4}Y  {hazards[i]:>8.4f}  {H_t:>8.4f}  {S_t:>8.4f}  {1-S_t:>8.4f}")

    print(f"\n  Bond Data")
    print(f"  {'Mat':>4}  {'Coupon':>8}  {'Face':>6}  {'Mkt Price':>10}")
    print("  " + "-" * 36)
    for b in bonds_data:
        print(f"  {b['maturity']:>4}Y  {b['coupon']:>8.4f}  {b['face']:>6.0f}  {b['market_price']:>10.4f}")

    print(f"\n  CDS Market Spreads")
    print(f"  {'Mat':>4}  {'Spread (bps)':>13}")
    print("  " + "-" * 22)
    for c in cds_data:
        print(f"  {c['maturity']:>4}Y  {c['spread_bps']:>13.2f}")

    print()


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic bond + CDS market data")
    p.add_argument("--rating",   default="BBB",
                   choices=list(RATING_HAZARD.keys()),
                   help="Credit rating (default: BBB)")
    p.add_argument("--sector",   default="industrials",
                   choices=list(SECTOR_MULTIPLIER.keys()),
                   help="Sector (default: industrials)")
    p.add_argument("--scenario", default="normal",
                   choices=list(SCENARIO_MULTIPLIER.keys()),
                   help="Market scenario (default: normal)")
    p.add_argument("--seed",     default=42, type=int,
                   help="Random seed for noise (default: 42)")
    p.add_argument("--out",      default=".",
                   help="Output directory for CSVs (default: current dir)")
    return p.parse_args()


def main():
    args = parse_args()

    rating   = args.rating.upper()
    sector   = args.sector.lower()
    scenario = args.scenario.lower()

    bonds_data, cds_data = generate(rating, sector, scenario, seed=args.seed)
    print_summary(rating, sector, scenario, bonds_data, cds_data)

    os.makedirs(args.out, exist_ok=True)
    bonds_path, cds_path = save_csvs(bonds_data, cds_data, args.out)

    print(f"  Saved → {bonds_path}")
    print(f"  Saved → {cds_path}")
    print(f"\n  Run the engine:")
    print(f"  python credit_basis_engine.py --bonds {bonds_path} --cds {cds_path}")


if __name__ == "__main__":
    main()
