#pragma once

class YieldCurve{

private:
    double r;

public:
    YieldCurve(double rate);

    double getRate(double t)const;

    double integral(double T)const;

};