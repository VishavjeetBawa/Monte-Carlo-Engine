#pragma once

#include "OptionParams.hpp"

namespace urop {

struct GPUParams {

    double S0;
    double K;
    double r;
    double sigma;

    int N;
    long long M;

    double dt;
    double discount;

    GPUParams(const AOP& p)
        : S0(p.S0_),
          K(p.K_),
          r(p.R_),
          sigma(p.sigma_),
          N(p.N_),
          M(p.M_),
          dt(p.dT_),
          discount(p.dF_) {}

};

}

