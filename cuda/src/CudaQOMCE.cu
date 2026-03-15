#include "CudaQOMCE.hpp"
#include "joe_kuo_sobol_data.hpp"
#include "RunStats.hpp"

#include <cuda_runtime.h>

#include <vector>

extern __constant__ unsigned int SOBOL_DIR[512][32];

__global__
void asian_qmc_kernel(
        urop::GPUParams,
        double*);

namespace urop {

CudaQOMCE::CudaQOMCE(const AOP& params)
    : gpu_params_(params)
{
    cudaMemcpyToSymbol(
        SOBOL_DIR,
        sobol_directions,
        sizeof(unsigned int) * 512 * 32
    );
}

MCResult CudaQOMCE::run()
{

    long long M = gpu_params_.M;

    double* d_results;

    cudaMalloc(&d_results, sizeof(double) * M);

    int threads = 256;

    int blocks =
        (M + threads - 1) / threads;

    asian_qmc_kernel<<<blocks, threads>>>(
        gpu_params_,
        d_results
    );

    std::vector<double> h_results(M);

    cudaMemcpy(
        h_results.data(),
        d_results,
        sizeof(double) * M,
        cudaMemcpyDeviceToHost
    );

    cudaFree(d_results);

    RunStats stats;

    for(double x : h_results)
        stats.update(x);

    MCResult result;

    result.price =
        stats.get_mean() * gpu_params_.discount;

    result.stderr =
        stats.get_std_error() * gpu_params_.discount;

    return result;
}

}
