#pragma once
#include "HazardCurve.hpp"
#include "YieldCurve.hpp"

class CDSPricer {
public:

    // Compute fair CDS spread
    static double fairSpread(
        const HazardCurve& hc,
        const YieldCurve& yc,
        double maturity,
        double recoveryRate,
        int steps = 1000
    );

private:

    static double survivalProbability(
        const HazardCurve& hc,
        double t
    );
};