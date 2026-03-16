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
        double* arith,
        double* geo)
{

    int path =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(path >= params.M) return;

    double z[MAX_STEPS];
    double w[MAX_STEPS];

    // Sobol normals
    for(int i=0;i<params.N;i++)
    {
        double u = sobol_sample(path,i);
        z[i] = inverse_normal(u);
    }

    brownian_bridge(z,w,params.N,params.dt);

    double logS = log(params.S0);

    double drift =
        (params.r - 0.5*params.sigma*params.sigma)
        * params.dt;

    double vol =
        params.sigma;

    double sum_arith = 0.0;
    double sum_geo = 0.0;


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
        double* arith,
        double* geo)
{

    int path =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(path >= params.M) return;

    double z[MAX_STEPS];
    double w[MAX_STEPS];

    // Sobol normals
    for(int i=0;i<params.N;i++)
    {
        double u = sobol_sample(path,i);
        z[i] = inverse_normal(u);
    }

    brownian_bridge(z,w,params.N,params.dt);

    double logS = log(params.S0);

    double drift =
        (params.r - 0.5*params.sigma*params.sigma)
        * params.dt;

    double vol =
        params.sigma;

    double sum_arith = 0.0;
    double sum_geo = 0.0;

    for(int i=0;i<params.N;i++)
    {
        double S = exp(logS);

        if(!isfinite(S))
        {
            S = params.K;   // fallback
        }

        sum_arith += S;
        sum_geo += logS;
    }

    double arith_avg = sum_arith / params.N;
    double geo_avg = exp(sum_geo/params.N);

    if(isnan(arith_avg))
    {
        printf("NaN detected at path %d\n", path);
    }

    arith[path] = fmax(arith_avg-params.K,0.0);
    geo[path] = fmax(geo_avg-params.K,0.0);
}

}



    double arith_avg = sum_arith / params.N;
    double geo_avg = exp(sum_geo/params.N);

    if(isnan(arith_avg))
    {
        printf("NaN detected at path %d\n", path);
    }

    arith[path] = fmax(arith_avg-params.K,0.0);
    geo[path] = fmax(geo_avg-params.K,0.0);
}

}
