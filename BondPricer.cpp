#include "BondPricer.hpp"
#include <cmath>

double BondPricer::price(const Bond& bond,
                         const YieldCurve& yc,
                         const HazardCurve& hc,
                         double recoveryRate) {

    double price = 0.0;
    double L = 1.0 - recoveryRate;

    for (double t : bond.paymentTimes) {

        double rInt = yc.integral(t);
        double hInt = hc.integral(t);

        double discount = std::exp(-(rInt + L * hInt));

        // Coupon payment
        price += bond.coupon * discount;
    }

    // Principal repayment
    double T = bond.paymentTimes.back();
    double rInt = yc.integral(T);
    double hInt = hc.integral(T);

    price += bond.face * std::exp(-(rInt + L * hInt));

    return price;
}