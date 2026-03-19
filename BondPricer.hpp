#pragma once
#include "Bond.hpp"
#include "HazardCurve.hpp"
#include "YieldCurve.hpp"

class BondPricer {
public:
    static double price(const Bond& bond,
                        const YieldCurve& yc,
                        const HazardCurve& hc,
                        double recoveryRate);
};