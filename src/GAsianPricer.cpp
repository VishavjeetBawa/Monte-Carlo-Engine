#include "GAsianPricer.hpp"
#include<cmath>

namespace urop {

double geo_pricer::price() const {
    const double sigma = params_.sigma_;
    const double r = params_.R_;
    const double T = params_.T_;
    const double K = params_.K_;
    const double S0 = params_.S0_;
    const int N = params_.N_;

    // variance of log G
    const double vG =
        sigma * sigma * T * (N + 1.0) * (2.0 * N + 1.0) / (6.0 * N * N);

    const double sqrt_vG = std::sqrt(vG);

    // mean of log G
    const double mG =
        std::log(S0) +
        (r - 0.5 * sigma * sigma) * T * (N + 1.0) / (2.0 * N);

    const double d1 = (mG - std::log(K) + vG) / sqrt_vG;
    const double d2 = d1 - sqrt_vG;

    return (std::exp(mG + 0.5 * vG) * normal_cdf(d1) - K * normal_cdf(d2)); //undiscounted multiply with exp(-r*t) for it to be discounted
}

}

