#pragma once

#include <cmath>
#include"OptionParams.hpp"

namespace urop{

class geo_pricer{
public:
    geo_pricer(const AOP& params):params_(params){}
    double price() const;
private:
    const AOP params_;

    static inline double normal_cdf(double x) { // cdf = control distributive fxn
        return 0.5 * std::erfc(-x/std::sqrt(2));
    }
};

}
