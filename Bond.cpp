#include "Bond.hpp"

Bond::Bond(double f,double c,const std::vector<double>& t)
    :face(f),coupon(c),paymentTimes(t){}