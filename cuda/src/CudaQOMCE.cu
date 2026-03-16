#include "CudaQOMCE.hpp"
#include "joe_kuo_sobol_data.hpp"
#include "RunStats.hpp"

#include <cuda_runtime.h>
#include <vector>

extern __constant__ unsigned int SOBOL_DIR[512][31];

namespace urop {

__global__
void asian_qmc_kernel(GPUParams,double*,double*);

CudaQOMCE::CudaQOMCE(const AOP& params)
    : gpu_params_(params)
{

cudaMemcpyToSymbol(
    SOBOL_DIR,
    (const unsigned int*)sobol_data::kDirectionNumbers,
    sizeof(unsigned int)*512*31
);

}

double normal_cdf(double x)
{
    return 0.5 * erfc(-x / std::sqrt(2.0));
}

double analytic_geometric_asian(const urop::GPUParams& p)
{
    double sigma_sq = p.sigma * p.sigma;

    double sigma_hat =
        p.sigma *
        std::sqrt((p.N + 1.0) * (2.0 * p.N + 1.0) /
                  (6.0 * p.N * p.N));

    double mu_hat =
        (p.r - 0.5 * sigma_sq) *
        (p.N + 1.0) / (2.0 * p.N)
        + 0.5 * sigma_hat * sigma_hat;

    double T = p.N * p.dt;

    double d1 =
        (std::log(p.S0 / p.K) +
        (mu_hat + 0.5 * sigma_hat * sigma_hat) * T)
        / (sigma_hat * std::sqrt(T));

    double d2 = d1 - sigma_hat * std::sqrt(T);

    double price =
        std::exp(-p.r * T) *
        (p.S0 * std::exp(mu_hat * T) * normal_cdf(d1)
        - p.K * normal_cdf(d2));

    return price;
}

MCResult CudaQOMCE::run()
{

    long long M = gpu_params_.M;

    double* d_arith;
    double* d_geo;

    cudaMalloc(&d_arith,sizeof(double)*M);
    cudaMalloc(&d_geo,sizeof(double)*M);

    int threads = 256;
    int blocks = (M+threads-1)/threads;

    asian_qmc_kernel<<<blocks,threads>>>(
        gpu_params_,
        d_arith,
        d_geo);

    cudaDeviceSynchronize();

    std::vector<double> h_arith(M);
    std::vector<double> h_geo(M);

    cudaMemcpy(
        h_arith.data(),
        d_arith,
        sizeof(double)*M,
        cudaMemcpyDeviceToHost);

    cudaMemcpy(
        h_geo.data(),
        d_geo,
        sizeof(double)*M,
        cudaMemcpyDeviceToHost);

    cudaFree(d_arith);
    cudaFree(d_geo);

    RunStats stats;
    BiRunStats cv;

    for(long long i=0;i<M;i++)
        cv.update(h_arith[i],h_geo[i]);

    
//BUG FIX
    double beta = cv.beta();

    if(std::isnan(beta) || std::isinf(beta))
        beta = 0.0;

    double geo_exact =
        analytic_geometric_asian(gpu_params_);

    for(long long i=0;i<M;i++)
    {
        double corrected =
            h_arith[i]
            - beta*(h_geo[i]-geo_exact);

        stats.update(corrected);
    }

    MCResult result;

    result.price =
        stats.get_mean()*gpu_params_.discount;

    result.std_error =
        stats.get_std_error()*gpu_params_.discount;

    return result;
}

}
