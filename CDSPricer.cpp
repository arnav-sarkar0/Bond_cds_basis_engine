#include "CDSPricer.hpp"
#include <cmath>

double CDSPricer::survivalProbability(
    const HazardCurve& hc,
    double t
) {
    return std::exp(-hc.integral(t));
}

double CDSPricer::fairSpread(
    const HazardCurve& hc,
    const YieldCurve& yc,
    double maturity,
    double recoveryRate,
    int steps
) {

    double dt = maturity / steps;
    double L = 1.0 - recoveryRate;

    double premiumLeg = 0.0;
    double protectionLeg = 0.0;

    for (int i = 1; i <= steps; i++) {

        double t = i * dt;

        double rInt = yc.integral(t);
        double hInt = hc.integral(t);

        double discount = std::exp(-rInt);
        double survival = std::exp(-hInt);

        double hazard = hc.getHazard(t);

        // Premium leg: s * ∫ discount * survival dt
        premiumLeg += discount * survival * dt;

        // Protection leg: L * ∫ discount * hazard * survival dt
        protectionLeg += discount * hazard * survival * dt;
    }

    // Fair spread = protection / premium
    return protectionLeg / premiumLeg;
}