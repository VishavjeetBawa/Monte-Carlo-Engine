#include <math.h>
#include "GPUParams.hpp"

extern __device__ double sobol_sample(unsigned int,int);
extern __device__ double inverse_normal(double);
extern __device__ void brownian_bridge(double*,double*,int,double);

#define MAX_STEPS 512

namespace urop {

__global__
void asian_qmc_kernel(
        GPUParams params,
        double* results)
{

    int path =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(path >= params.M) return;

    double z[MAX_STEPS];
    double w[MAX_STEPS];

    for(int i=0;i<params.N;i++)
    {
        double u = sobol_sample(path+1,i);
        z[i] = inverse_normal(u);
    }

    brownian_bridge(z,w,params.N,params.dt);

    double logS = log(params.S0);

    double drift =
        (params.r - 0.5*params.sigma*params.sigma)
        * params.dt;

    double vol =
        params.sigma;

    double sum = 0.0;

    for(int i=0;i<params.N;i++)
    {
        logS += drift + vol * w[i];

        double S = exp(logS);

        sum += S;
    }

    double avg = sum / params.N;

    double payoff = fmax(avg - params.K,0.0);

    results[path] = payoff;
}

}
