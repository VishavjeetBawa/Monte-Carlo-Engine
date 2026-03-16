#include "CudaQOMCE.hpp"
#include "joe_kuo_sobol_data.hpp"
#include "RunStats.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace urop {

// Global pointer to bypass constant memory visibility issues during debug
static unsigned int* d_debug_sobol_ptr = nullptr;

// --- Hardened Kernel ---
__global__ void asian_qmc_debug_kernel(GPUParams params, double* arith, double* geo, unsigned int* sobol_dirs) {
    long long path = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (path >= params.M) return;

    double z[128]; 
    double w[128];

    // 1. Sobol Generation with logic checks
    unsigned int g = (unsigned int)path ^ ((unsigned int)path >> 1);
    bool all_zeros = true;

    for (int i = 0; i < params.N; i++) {
        unsigned int x = 0;
        for (int b = 0; b < 31; b++) {
            unsigned int dir = sobol_dirs[i * 31 + b];
            if (dir > 0) all_zeros = false;
            if (g & (1u << b)) x ^= dir;
        }
        
        double u = ((double)x + 0.5) * 2.3283064365386963e-10;
        u = fmax(1e-10, fmin(u, 1.0 - 1e-10)); 
        z[i] = inverse_normal(u);
    }

    // DEBUG PRINT for the very first thread
    if (path == 0) {
        printf("[Kernel Path 0] Sobol All Zeros: %s, First Z: %f\n", 
               all_zeros ? "YES (ERROR)" : "NO (OK)", z[0]);
    }

    brownian_bridge(z, w, params.N, params.dt);

    double logS = log(params.S0);
    double drift = (params.r - 0.5 * params.sigma * params.sigma) * params.dt;
    double sum_arith = 0.0, sum_geo = 0.0;

    for (int i = 0; i < params.N; i++) {
        logS += drift + params.sigma * w[i];
        double S = exp(fmin(logS, 50.0)); 
        sum_arith += S;
        sum_geo += logS;
    }

    arith[path] = fmax((sum_arith / params.N) - params.K, 0.0);
    geo[path] = fmax(exp(sum_geo / params.N) - params.K, 0.0);
}

// --- Host Implementation ---

CudaQOMCE::CudaQOMCE(const AOP& params) : gpu_params_(params) {
    size_t sz = sizeof(unsigned int) * 512 * 31;
    
    // 1. Host side check: Is the header data valid?
    if (sobol_data::kDirectionNumbers[0][0] == 0 && sobol_data::kDirectionNumbers[1][1] == 0) {
        printf("[DEBUG ERROR] Host Sobol data appears to be unitialized/zero!\n");
    }

    // 2. Allocate Global Memory for Sobol (Bypasses __constant__ scope issues)
    if (d_debug_sobol_ptr == nullptr) {
        cudaMalloc(&d_debug_sobol_ptr, sz);
        cudaError_t err = cudaMemcpy(d_debug_sobol_ptr, sobol_data::kDirectionNumbers, sz, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) printf("[DEBUG ERROR] Sobol Memcpy failed: %s\n", cudaGetErrorString(err));
    }
}

MCResult CudaQOMCE::run() {
    cudaDeviceSetLimit(cudaLimitStackSize, 32768); 

    long long M = gpu_params_.M;
    double *d_arith, *d_geo;
    cudaMalloc(&d_arith, sizeof(double) * M);
    cudaMalloc(&d_geo, sizeof(double) * M);
    cudaMemset(d_arith, 0, sizeof(double) * M);
    cudaMemset(d_geo, 0, sizeof(double) * M);

    int threads = 64; 
    int blocks = (int)((M + threads - 1) / threads);

    printf("\n--- GPU Debug Run Start ---\n");
    printf("Params: S0=%.2f, K=%.2f, N=%d, Threads=%d, Blocks=%d\n", 
            gpu_params_.S0, gpu_params_.K, gpu_params_.N, threads, blocks);

    asian_qmc_debug_kernel<<<blocks, threads>>>(gpu_params_, d_arith, d_geo, d_debug_sobol_ptr);
    
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) printf("[DEBUG ERROR] Kernel Launch: %s\n", cudaGetErrorString(launch_err));

    cudaDeviceSynchronize();

    std::vector<double> h_arith(M), h_geo(M);
    cudaMemcpy(h_arith.data(), d_arith, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_geo.data(), d_geo, sizeof(double) * M, cudaMemcpyDeviceToHost);

    // 3. Post-Kernel Analysis
    int zero_payoffs = 0;
    for(int i=0; i<std::min((long long)1000, M); i++) if(h_arith[i] <= 0.0) zero_payoffs++;
    printf("Sample Check: %d out of first 1000 paths have 0.0 payoff\n", zero_payoffs);

    BiRunStats cv;
    long long valid = 0;
    for (long long i = 0; i < M; i++) {
        if (std::isfinite(h_arith[i]) && std::isfinite(h_geo[i])) {
            cv.update(h_arith[i], h_geo[i]);
            valid++;
        }
    }

    double beta = (valid > 1) ? cv.beta() : 0.0;
    if (!std::isfinite(beta)) {
        printf("[DEBUG WARNING] Beta is NaN. Setting to 0.0. Valid paths: %lld\n", valid);
        beta = 0.0;
    }

    double geo_exact = analytic_geometric_asian(gpu_params_);
    RunStats stats;
    for (long long i = 0; i < M; i++) {
        double corrected = h_arith[i] - beta * (h_geo[i] - geo_exact);
        if (std::isfinite(corrected)) stats.update(corrected);
    }

    cudaFree(d_arith); cudaFree(d_geo);
    printf("--- GPU Debug Run End ---\n\n");

    return {stats.get_mean() * gpu_params_.discount, stats.get_std_error() * gpu_params_.discount};
}

} // namespace urop
