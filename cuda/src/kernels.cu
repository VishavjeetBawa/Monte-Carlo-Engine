#include <math.h>
#include "GPUParams.hpp"

extern __device__ double sobol_sample(unsigned int,int);
extern __device__ double inverse_normal(double);

__device__
double brownian_bridge(double prev, double z)
{
    return z + 0.5 * prev;
}

__global__
void asian_qmc_kernel(
        urop::GPUParams params,
        double* results)
{

    int path =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(path >= params.M) return;

    double logS = log(params.S0);

    double drift =
        (params.r - 0.5 * params.sigma * params.sigma)
        * params.dt;

    double vol =
        params.sigma * sqrt(params.dt);

    double running_sum = 0.0;

    double prev_z = 0.0;

    for(int t = 0; t < params.N; ++t)
    {
        double u = sobol_sample(path, t);

        double z = inverse_normal(u);

        z = brownian_bridge(prev_z, z);

        prev_z = z;

        logS += drift + vol * z;

        double S = exp(logS);

        running_sum += S;
    }

    double avg = running_sum / params.N;

    double payoff = fmax(avg - params.K, 0.0);

    results[path] = payoff;
}
