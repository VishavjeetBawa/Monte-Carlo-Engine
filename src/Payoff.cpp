#include "Payoff.hpp"
#include<algorithm>
#include<vector>
#include <numeric>
#include <cmath>

namespace urop{

double AsianCallPayoff::calculate(const std::vector<double>& path_prices) const {
    if(path_prices.empty()){
        return 0.0;
    }

    double sum = std::accumulate(path_prices.begin(), path_prices.end() , 0.0);
    double avg = sum/path_prices.size();
    return std::max(avg - K_,0.0);
}

double GeometricAsianPayoff::calculate(const std::vector<double>& path_prices) const{
    if(path_prices.empty()){
        return 0.0;
    }

    double sum_log = 0.0;
    for(double s : path_prices){
        sum_log += std::log(s);
    }
    double geom_mean = std::exp(sum_log/path_prices.size());
    return std::max(0.0 , geom_mean-K_);
}

}
