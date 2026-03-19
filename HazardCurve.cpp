#include "HazardCurve.hpp"

HazardCurve::HazardCurve(const std::vector<double>& t,
                         const std::vector<double>& h)
    : times(t), hazards(h) {}

double HazardCurve::getHazard(double t) const {
    for (size_t i = 0; i < times.size(); i++) {
        if (t <= times[i])
            return hazards[i];
    }
    return hazards.back();
}

double HazardCurve::integral(double T) const {
    double result = 0.0;
    double prev = 0.0;

    for (size_t i = 0; i < times.size(); i++) {
        double dt = std::min(T, times[i]) - prev;
        if (dt > 0)
            result += hazards[i] * dt;

        prev = times[i];
        if (times[i] >= T) break;
    }

    return result;
}