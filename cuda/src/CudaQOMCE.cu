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

    double beta = cv.beta();

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
