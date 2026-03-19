#include "YieldCurve.hpp"

YieldCurve::YieldCurve(double rate):r(rate){}

double YieldCurve::getRate(double t)const{
    return r;
}

double YieldCurve::integral(double T)const{
    return r*T;
     
}