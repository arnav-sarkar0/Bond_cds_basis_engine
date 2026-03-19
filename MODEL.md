# Model Documentation

## Duffie–Singleton Reduced-Form Credit Model

### Overview

This engine implements the Duffie–Singleton (1999) reduced-form credit model. In this framework, default is modelled as the first jump of a Poisson process with intensity $h(t)$ — the **hazard rate**.

The key insight of Duffie–Singleton is that a defaultable bond can be priced like a default-free bond by replacing the risk-free discount rate $r(t)$ with an **adjusted rate**:

$$R(t) = r(t) + h(t) \cdot L$$

where $L = 1 - R_{\text{recovery}}$ is the fractional loss given default.

---

## Hazard Rate Curve

We assume a **piecewise-constant** hazard rate term structure:

$$h(t) = h_i \quad \text{for } t \in (t_{i-1}, t_i]$$

This gives a simple closed-form for the cumulative hazard:

$$H(t) = \int_0^t h(s)\, ds = \sum_{i: t_i \leq t} h_i \cdot \Delta t_i + h_k \cdot (t - t_{k-1})$$

where $k$ is the interval containing $t$.

---

## Bond Pricing

For a coupon bond with payment times $\{t_1, \ldots, t_T\}$, face value $F$, coupon $C$:

$$P = \sum_{i=1}^{T} C \cdot D(t_i) + F \cdot D(T)$$

where the **risky discount factor** is:

$$D(t) = e^{-(r \cdot t + L \cdot H(t))}$$

This combines risk-free discounting and credit-adjusted discounting in a single exponential. Note that this uses $L \cdot H(t)$ — not the full hazard — reflecting the RMV (recovery of market value) assumption.

---

## CDS Pricing

A CDS contract exchanges:
- **Premium leg**: buyer pays spread $s$ periodically while the reference entity survives
- **Protection leg**: seller pays $L \cdot \text{Notional}$ upon default

The **fair spread** sets the two legs equal:

$$s = \frac{L \int_0^T e^{-r t} \cdot h(t) \cdot Q(\tau > t)\, dt}{\int_0^T e^{-r t} \cdot Q(\tau > t)\, dt}$$

where the **survival probability** is:

$$Q(\tau > t) = e^{-H(t)}$$

Note the asymmetry: CDS survival uses the **full hazard** $e^{-H(t)}$, while bond discounting uses $e^{-L \cdot H(t)}$. This is correct and intentional.

Both integrals are computed numerically with 1000 steps (configurable).

---

## Calibration

### Bootstrap

Hazard rates are calibrated one maturity at a time. For maturity $t_i$, we solve:

$$P_{\text{model}}(h_1, \ldots, h_i) = P_{\text{market},i}$$

holding $h_1, \ldots, h_{i-1}$ fixed. This is a 1D root-finding problem solved with **Brent's method** (`scipy.optimize.brentq`), which is guaranteed to converge for a continuous objective on a bracketed interval.

### Smoothing (optional)

Raw bootstrapped hazards can be jagged due to market noise. An optional smoothing step minimises:

$$\min_h \sum_i \left(P_{\text{model},i}(h) - P_{\text{market},i}\right)^2 + \lambda \sum_j (h_{j+1} - h_j)^2$$

The second term is a **roughness penalty** that discourages large jumps between adjacent hazard rates. Solved with **L-BFGS-B** (bounded quasi-Newton). Default penalty $\lambda = 10$.

---

## Basis Detection

The **Bond–CDS basis** is:

$$\text{Basis} = s^{\text{market}} - s^{\text{model}}$$

A non-zero basis indicates the two markets are pricing default risk differently.

### Why Does Basis Exist?

In theory, the basis should be zero (no arbitrage). In practice:

| Factor | Effect on Basis |
|--------|----------------|
| Funding costs / repo | Positive basis (CDS appears cheap) |
| Bond illiquidity | Positive basis in stress |
| CDS demand (hedging) | Negative basis |
| Technical supply/demand | Either direction |
| Recovery disagreement | Depends on recovery assumption |

### The Negative Basis Trade

When basis < 0 (bonds cheap vs CDS):
- **Buy the bond** (cheap credit)
- **Buy CDS protection** (hedge default risk)
- **Net P&L**: carry the coupon minus CDS premium, profit when basis converges

This is the classic "negative basis trade" run by credit hedge funds.

---

## Assumptions & Limitations

| Assumption | Impact | How to Relax |
|------------|--------|--------------|
| Flat yield curve | Model risk for longer maturities | Bootstrap swap curve from LIBOR/SOFR |
| Constant recovery | Can distort hazard rates | Use market-implied recovery |
| Piecewise-constant $h(t)$ | Jagged term structure | Nelson-Siegel parameterisation |
| No liquidity premium | Hazard absorbs liquidity | Add $\ell(t)$ spread component |
| Continuous coupons | Minor pricing error | Add day count conventions |

---

## References

1. Duffie, D. & Singleton, K. (1999). *Modeling Term Structures of Defaultable Bonds*. Review of Financial Studies, 12(4), 687–720.

2. O'Kane, D. (2008). *Modelling Single-name and Multi-name Credit Derivatives*. Wiley Finance.

3. Blanco, R., Brennan, S. & Marsh, I. (2005). *An Empirical Analysis of the Dynamic Relation between Investment-Grade Bonds and Credit Default Swaps*. Journal of Finance, 60(5), 2255–2281.

4. Lando, D. (2004). *Credit Risk Modeling: Theory and Applications*. Princeton University Press.
