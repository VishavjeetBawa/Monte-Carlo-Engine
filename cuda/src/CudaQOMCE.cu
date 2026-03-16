#include "CudaQOMCE.hpp"
#include "joe_kuo_sobol_data.hpp"
#include "RunStats.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace urop {

static unsigned int* d_debug_sobol_ptr = nullptr;

// --- Math Utilities ---

__device__ double inverse_normal(double u) {
    const double a1 = -39.6968302866538, a2 = 220.946098424521, a3 = -275.928510446969;
    const double a4 = 138.357751867269, a5 = -30.6647980661472, a6 = 2.50662827745924;
    const double b1 = -54.4760987982241, b2 = 161.585836858041, b3 = -155.698979859887;
    const double b4 = 66.8013118877197, b5 = -13.2806815528857;
    const double c1 = -0.00778489400243029, c2 = -0.322396458041136, c3 = -2.40075827716184;
    const double c4 = -2.54973253934373, c5 = 4.37466414146497, c6 = 2.93816398269878;
    const double d1 = 0.00778469570904146, d2 = 0.32246712907004, d3 = 2.445134137143, d4 = 3.75440866190742;

    double q, r;
    if (u < 0.02425) {
        q = sqrt(-2.0 * log(u));
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }
    if (u > 0.97575) {
        q = sqrt(-2.0 * log(1.0 - u));
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }
    q = u - 0.5; r = q * q;
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
}

__device__ void brownian_bridge(double* z, double* w, int N, double dt) {
    w[0] = 0.0;
    w[N - 1] = sqrt((double)N * dt) * z[0];
    int left[128], right[128]; 
    int top = 0, dim = 1;
    left[top] = 0; right[top] = N - 1;

    while (top >= 0 && dim < N) {
        int l = left[top], r = right[top];
        top--;
        if (r - l <= 1) continue;
        int m = (l + r) / 2;
        double t_l = l * dt, t_r = r * dt, t_m = m * dt;
        double mean = ((t_r - t_m) * w[l] + (t_m - t_l) * w[r]) / (t_r - t_l);
        double var = fmax((t_m - t_l) * (t_r - t_m) / (t_r - t_l), 1e-14);
        w[m] = mean + sqrt(var) * z[dim++];
        if (top + 1 < 128) { // Guard stack
            top++; left[top] = l; right[top] = m;
            top++; left[top] = m; right[top] = r;
        }
    }
    for (int i = N - 1; i > 0; i--) w[i] = w[i] - w[i - 1];
    w[0] = sqrt(dt) * z[N - 1];
}

// --- Debug Kernel ---

__global__ void asian_qmc_debug_kernel(GPUParams params, double* arith, double* geo, unsigned int* sobol_dirs) {
    long long path = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (path >= params.M) return;

    double z[128], w[128];

    // 1. Sobol + Inverse Normal
    unsigned int g = (unsigned int)path ^ ((unsigned int)path >> 1);
    for (int i = 0; i < params.N; i++) {
        unsigned int x = 0;
        for (int b = 0; b < 31; b++) {
            if (g & (1u << b)) x ^= sobol_dirs[i * 31 + b];
        }
        double u = ((double)x + 0.5) * 2.3283064365386963e-10;
        z[i] = inverse_normal(fmax(1e-10, fmin(u, 1.0 - 1e-10)));
    }

    brownian_bridge(z, w, params.N, params.dt);

    // 2. Pricing
    double logS = log(params.S0);
    double drift = (params.r - 0.5 * params.sigma * params.sigma) * params.dt;
    double sum_arith = 0.0, sum_geo = 0.0;

    for (int i = 0; i < params.N; i++) {
        logS += drift + params.sigma * w[i];
        double S = exp(fmin(logS, 50.0)); // Overflow guard
        sum_arith += S;
        sum_geo += logS;
    }

    arith[path] = fmax((sum_arith / params.N) - params.K, 0.0);
    geo[path] = fmax(exp(sum_geo / params.N) - params.K, 0.0);

    // CRITICAL DEBUG: Print path 0 to see what's happening
    if (path == 0) {
        printf("[GPU Thread 0] Arith: %f, Geo: %f, Params.N: %d, S0: %f\n", arith[0], geo[0], params.N, params.S0);
    }
}

// --- Host Code ---

CudaQOMCE::CudaQOMCE(const AOP& params) : gpu_params_(params) {
    size_t sz = sizeof(unsigned int) * 512 * 31;
    if (d_debug_sobol_ptr == nullptr) {
        cudaMalloc(&d_debug_sobol_ptr, sz);
        cudaMemcpy(d_debug_sobol_ptr, sobol_data::kDirectionNumbers, sz, cudaMemcpyHostToDevice);
    }
}

double normal_cdf(double x) { return 0.5 * erfc(-x / std::sqrt(2.0)); }

double analytic_geometric_asian(const urop::GPUParams& p) {
    double T = p.N * p.dt;
    double sigma_hat = p.sigma * std::sqrt((p.N + 1.0) * (2.0 * p.N + 1.0) / (6.0 * p.N * p.N));
    double mu_hat = (p.r - 0.5 * p.sigma * p.sigma) * (p.N + 1.0) / (2.0 * p.N) + 0.5 * sigma_hat * sigma_hat;
    double d1 = (std::log(p.S0 / p.K) + (mu_hat + 0.5 * sigma_hat * sigma_hat) * T) / (sigma_hat * std::sqrt(T));
    double d2 = d1 - sigma_hat * std::sqrt(T);
    double price = std::exp(-p.r * T) * (p.S0 * std::exp(mu_hat * T) * normal_cdf(d1) - p.K * normal_cdf(d2));
    return std::isfinite(price) ? price : 0.0;
}

MCResult CudaQOMCE::run() {
    cudaDeviceSetLimit(cudaLimitStackSize, 32768); 

    long long M = gpu_params_.M;
    double *d_arith, *d_geo;
    cudaMalloc(&d_arith, sizeof(double) * M);
    cudaMalloc(&d_geo, sizeof(double) * M);

    int threads = 64; 
    int blocks = (int)((M + threads - 1) / threads);

    asian_qmc_debug_kernel<<<blocks, threads>>>(gpu_params_, d_arith, d_geo, d_debug_sobol_ptr);
    cudaDeviceSynchronize();

    std::vector<double> h_arith(M), h_geo(M);
    cudaMemcpy(h_arith.data(), d_arith, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_geo.data(), d_geo, sizeof(double) * M, cudaMemcpyDeviceToHost);

    // Reduction with Beta guard
    BiRunStats cv;
    long long valid = 0;
    double var_y_check = 0.0;
    for (long long i = 0; i < M; i++) {
        if (std::isfinite(h_arith[i]) && std::isfinite(h_geo[i])) {
            cv.update(h_arith[i], h_geo[i]);
            var_y_check += h_geo[i] * h_geo[i]; // Heuristic variance check
            valid++;
        }
    }

    // Force beta to 0 if we have no statistical variance or valid paths
    double beta = (valid > 1 && var_y_check > 1e-12) ? cv.beta() : 0.0;
    if (!std::isfinite(beta)) beta = 0.0;

    double geo_exact = analytic_geometric_asian(gpu_params_);
    RunStats stats;
    for (long long i = 0; i < M; i++) {
        double corrected = h_arith[i];
        if (std::isfinite(h_arith[i]) && std::isfinite(h_geo[i])) {
            corrected = h_arith[i] - beta * (h_geo[i] - geo_exact);
        }
        if (std::isfinite(corrected)) stats.update(corrected);
    }

    cudaFree(d_arith); cudaFree(d_geo);
    return {stats.get_mean() * gpu_params_.discount, stats.get_std_error() * gpu_params_.discount};
}

} // namespace urop
