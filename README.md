# Bond–CDS Basis Engine

> A quantitative credit risk system that extracts default probabilities from corporate bond prices, prices CDS contracts using the same hazard curve, and detects basis arbitrage opportunities between bond and CDS markets.

Built on the **Duffie–Singleton** reduced-form credit model with a C++ pricing engine and Python calibration/analysis layer.

---

## What This Does

Most market participants look at bonds *or* CDS in isolation. This engine looks at both simultaneously:

```
Corporate Bond Prices
        │
        ▼
  Calibrate h(t)          ← piecewise-constant hazard rate curve
  [Duffie–Singleton]
        │
        ▼
  Price CDS using          ← same h(t), consistent framework
  same h(t)
        │
        ▼
  Compare with             ← market-quoted CDS spreads
  Market CDS
        │
        ▼
  Basis = Market CDS − Model CDS
  Generate trade signals
```

The core insight: the same company has the same default risk whether you're looking at its bonds or its CDS. If the two markets disagree — that's a basis trade.

---

## Project Structure

```
BOND_CDS_ENGINE/
│
├── cpp/
│   ├── include/
│   │   ├── Bond.hpp
│   │   ├── BondPricer.hpp
│   │   ├── CDSPricer.hpp
│   │   ├── HazardCurve.hpp
│   │   └── YieldCurve.hpp
│   ├── src/
│   │   ├── Bond.cpp
│   │   ├── BondPricer.cpp
│   │   ├── CDSPricer.cpp
│   │   ├── HazardCurve.cpp
│   │   └── YieldCurve.cpp
│   ├── CMakeLists.txt
│   └── main.cpp
│
├── python/
│   ├── credit_basis_engine.py   ← core calibration + basis engine
│   ├── dashboard.py             ← Streamlit dashboard
│   └── generate_market_data.py  ← synthetic data generator
│
├── data/                        ← CSV inputs/outputs
├── docs/                        ← additional documentation
├── notebooks/                   ← Jupyter exploration
├── engine.exe                   ← compiled C++ engine (Windows)
└── README.md
```

---

## The Math

### Bond Pricing — Duffie–Singleton

$$P = \sum_{t=1}^{T} \text{CF}(t) \cdot e^{-(r \cdot t \; + \; L \cdot H(t))}$$

where:
- $H(t) = \int_0^t h(s)\, ds$ — cumulative hazard
- $L = 1 - R$ — loss given default
- $R$ — recovery rate (default 40%)
- $r$ — risk-free rate (flat yield curve)

### CDS Fair Spread

$$s = \frac{L \int_0^T e^{-rt} \cdot h(t) \cdot e^{-H(t)}\, dt}{\int_0^T e^{-rt} \cdot e^{-H(t)}\, dt} = \frac{\text{Protection Leg}}{\text{Premium Leg}}$$

### Survival Probability

$$Q(\tau > t) = e^{-H(t)}$$

> Note: Bond discounting uses $e^{-L \cdot H(t)}$ (RMV convention). CDS survival uses $e^{-H(t)}$ (full hazard). This distinction is critical and is preserved throughout.

### Basis

$$\text{Basis} = s^{\text{market}} - s^{\text{model}}$$

| Basis | Interpretation | Signal |
|-------|---------------|--------|
| > +5 bps | CDS expensive vs bonds | Buy Bond / Sell CDS protection |
| < −5 bps | CDS cheap vs bonds | Sell Bond / Buy CDS protection |
| ±5 bps | Fairly priced | No trade |

---

## C++ Engine

The pricing engine is written in C++ for performance. It implements four components:

| Class | Role |
|-------|------|
| `YieldCurve` | Flat risk-free rate, $r \cdot T$ integral |
| `HazardCurve` | Piecewise-constant $h(t)$, exact integral |
| `BondPricer` | Duffie–Singleton bond price |
| `CDSPricer` | Premium/protection legs, fair spread |

### Build

```bash
cd cpp
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

The compiled binary is called by the Python layer via subprocess, receiving hazard rates and maturity as command-line arguments and returning `bond_price cds_spread`.

---

## Python Layer

The Python layer mirrors the C++ math exactly so it can be used standalone (no compilation required). All three files are self-contained.

### `credit_basis_engine.py` — Core Engine

```bash
# Use built-in default data
python credit_basis_engine.py

# Load from CSV files
python credit_basis_engine.py --bonds data/bonds.csv --cds data/cds_market.csv

# Skip smoothing
python credit_basis_engine.py --bonds data/bonds.csv --cds data/cds_market.csv --no-smooth
```

**Pipeline:**
1. Bootstrap $h(t)$ from bond prices using Brent's method
2. Smooth the hazard curve via L-BFGS-B optimisation (optional)
3. Compute survival and default probabilities
4. Price CDS using calibrated $h(t)$
5. Compare with market CDS → compute basis
6. Output signal table + plots

### `generate_market_data.py` — Synthetic Data Generator

Generates realistic bond prices and CDS spreads based on credit rating, sector, and market scenario. Useful for testing and demonstration.

```bash
# BBB energy company in stress
python generate_market_data.py --rating BBB --sector energy --scenario stress

# B-rated financials in crisis  
python generate_market_data.py --rating B --sector financials --scenario crisis

# Full options
python generate_market_data.py --help
```

| Flag | Options | Default |
|------|---------|---------|
| `--rating` | AAA, AA, A, BBB, BB, B, CCC | BBB |
| `--sector` | financials, energy, tech, industrials, utilities | industrials |
| `--scenario` | normal, stress, crisis | normal |
| `--seed` | any integer | 42 |
| `--out` | output directory | . |

Outputs `bonds.csv` and `cds_market.csv` ready to feed into the engine.

### `dashboard.py` — Streamlit Dashboard

Interactive web dashboard wrapping the full engine.

```bash
pip install streamlit plotly pandas scipy numpy
streamlit run python/dashboard.py
```

**Features:**
- Configure rating/sector/scenario and generate data on the fly
- Upload your own `bonds.csv` and `cds_market.csv`
- Live engine run with real-time calibration
- Download generated CSVs
- Tabbed layout: Inputs · Results · Charts

---

## CSV Format

If you have real market data, drop it into these two files:

**`bonds.csv`**
```csv
maturity,coupon,face,market_price
1,0.0425,100,98.50
3,0.0500,100,94.20
5,0.0575,100,88.75
```

**`cds_market.csv`**
```csv
maturity,spread_bps
1,85.0
3,142.0
5,198.0
```

`coupon` and `face` are optional in `bonds.csv` — they fall back to the config defaults (5.0 and 100.0).

---

## Installation

```bash
git clone https://github.com/yourusername/bond-cds-basis-engine.git
cd bond-cds-basis-engine

# Python dependencies
pip install numpy scipy matplotlib pandas plotly streamlit

# Build C++ engine (optional — Python pricers work standalone)
cd cpp && mkdir build && cd build
cmake .. && cmake --build . --config Release
```

**Python version:** 3.10+  
**C++ standard:** C++17

---

## Example Output

```
[Phase 1] Bootstrapping hazard rates
  Mat    Coupon       h_i     Model P    Market P
  ──────────────────────────────────────────────
   1Y    0.0521    0.052100    95.0000    95.0000
   2Y    0.0614    0.071300    93.0000    93.0000
   3Y    0.0742    0.098200    90.0000    90.0000
   4Y    0.0891    0.134500    85.0000    85.0000
   5Y    0.1063    0.178900    80.0000    80.0000

── Bond–CDS Basis Table ──────────────────────────────────────────
  Mat   Model CDS   Market CDS      Basis  Signal
  ──────────────────────────────────────────────────────────────
   1Y    52.1 bp      68.4 bp      +16.3bp  Buy Bond / Sell CDS protection
   2Y    71.3 bp      59.8 bp      -11.5bp  Sell Bond / Buy CDS protection
   3Y    98.2 bp     121.0 bp      +22.8bp  Buy Bond / Sell CDS protection
   4Y   134.5 bp     128.3 bp       -6.2bp  Sell Bond / Buy CDS protection
   5Y   178.9 bp     201.4 bp      +22.5bp  Buy Bond / Sell CDS protection
```

---

## Real Data

With real market data this engine replicates what credit trading desks run daily. The basis reflects structural dislocations:

- **Funding/repo constraints** — dealers can't arb the basis when funding is tight
- **Liquidity mismatch** — CDS more liquid than cash bonds in stress
- **Supply/demand** — convexity hedging creates persistent CDS demand
- **Recovery disagreement** — bond vs CDS recovery assumptions differ

Real data sources: Bloomberg (SRCH for bonds, CDSD for CDS), Refinitiv, or university terminal access.

---

## Limitations & Assumptions

- Flat risk-free yield curve (extend `YieldCurve` for term structure)
- Constant recovery rate across maturities
- No accrued interest / day count conventions
- Piecewise-constant hazard (could extend to Nelson-Siegel)
- No liquidity premium decomposition
- Single-name only (no basket/index CDS)

These are intentional simplifications — each is a potential extension.

---

## Potential Extensions

- [ ] Term structure of interest rates (swap curve bootstrapping)
- [ ] Stochastic hazard rates (CIR / Vasicek)
- [ ] Liquidity premium decomposition
- [ ] Monte Carlo default simulation
- [ ] Index CDS (CDX / iTraxx)
- [ ] Live Bloomberg data feed
- [ ] Historical backtesting of basis signals

---

## References

- Duffie, D. & Singleton, K. (1999). *Modeling Term Structures of Defaultable Bonds*. Review of Financial Studies.
- O'Kane, D. (2008). *Modelling Single-name and Multi-name Credit Derivatives*. Wiley Finance.
- Blanco, R., Brennan, S. & Marsh, I. (2005). *An Empirical Analysis of the Dynamic Relation between Investment-Grade Bonds and Credit Default Swaps*. Journal of Finance.

---

## License

MIT License — see `LICENSE` for details.
