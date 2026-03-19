#pragma once
#include <vector>

class HazardCurve {
private:
    std::vector<double> times;     // breakpoints
    std::vector<double> hazards;   // hazard rates per interval

public:
    HazardCurve(const std::vector<double>& times,
                const std::vector<double>& hazards);

    double getHazard(double t) const;

    // Integral of h(t) from 0 to T
    double integral(double T) const;
};