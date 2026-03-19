#include <iostream>
#include <vector>
#include <cstdlib>
#include "CDSPricer.hpp"
#include "HazardCurve.hpp"
#include "YieldCurve.hpp"
#include "Bond.hpp"
#include "BondPricer.hpp"

int main(int argc, char* argv[]) {
if (argc < 3) {
    std::cerr << "Usage: engine h1 h2 ... hn maturity" << std::endl;
    return 1;
}
int n = argc - 2;  // last arg is maturity

std::vector<double> hazards;
for (int i = 1; i <= n; i++) {
    hazards.push_back(atof(argv[i]));
}

double maturity = atof(argv[argc - 1]);

std::vector<double> times;
for (int i = 1; i <= n; i++) {
    times.push_back((double)i);
}



HazardCurve hc(times, hazards);
YieldCurve yc(0.05);

// build payment schedule
std::vector<double> payments;
for (int i = 1; i <= maturity; i++) {
    payments.push_back(i);
}

Bond bond(100.0, 5.0, payments);

double price = BondPricer::price(bond, yc, hc, 0.4);

//std::cout << price << std::endl;

double cds = CDSPricer::fairSpread(
    hc,
    yc,
    maturity,
    0.4
);

std::cout << price << " " << cds << std::endl;


    return 0;
}