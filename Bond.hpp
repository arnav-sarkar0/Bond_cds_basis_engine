#pragma once
#include <vector>

class Bond{
public:
    double face;
    double coupon;
    std::vector<double> paymentTimes;

    Bond(double face,double coupon,const std::vector<double>&paymentTimes);
};