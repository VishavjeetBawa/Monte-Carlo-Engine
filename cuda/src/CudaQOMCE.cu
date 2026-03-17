#include "CudaQOMCE.hpp"
#include "joe_kuo_sobol_data.hpp"
#include "RunStats.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace urop {

static unsigned int* d_sobol_ptr = nullptr;

// --- Per‑path digital shift (deterministic) ---
__device__ unsigned int get_shift(long long path) {
    return (unsigned int)(path * 0x9e3779b97f4a7c15ULL) ^ 0xbf58476d1ce4e5b9ULL;
}

__device__ double inverse_normal(double u) {
    u = fmax(1e-10, fmin(u, 1.0 - 1e-10));
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

__device__ void brownian_bridge(double* z, int N, double dt) {
    double W[129];                // N+1 ≤ 129
    W[0] = 0.0;
    W[N] = sqrt(N * dt) * z[0];   // Brownian at final time

    int left[128], right[128];
    int top = 0;
    left[top] = 0;
    right[top] = N;
    int dim = 1;

    while (top >= 0 && dim < N) {
        int l = left[top], r = right[top];
        top--;
        if (r - l <= 1) continue;

        int m = (l + r) / 2;
        double tl = l * dt, tr = r * dt, tm = m * dt;
        double mean = ((tr - tm) * W[l] + (tm - tl) * W[r]) / (tr - tl);
        double var  = (tm - tl) * (tr - tm) / (tr - tl);
        if (var < 0.0) var = 0.0;
        W[m] = mean + sqrt(var) * z[dim];
        if (!isfinite(W[m])) W[m] = 0.0;
        dim++;

        left[++top] = l;  right[top] = m;
        left[++top] = m;  right[top] = r;
    }

    for (int i = 0; i < N; ++i) {
        double inc = (W[i+1] - W[i]) / sqrt(dt);
        z[i] = isfinite(inc) ? inc : 0.0;
    }
}

// Each thread processes one antithetic pair and stores the averaged payoffs
__global__ void asian_qmc_kernel(GPUParams params, double* arith, double* geo, unsigned int* sobol_dirs) {
    long long pair = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (pair >= params.M) return;          // params.M is now number of pairs

    double z[128];
    // Sobol index for the first path of the pair
    unsigned int idx = (unsigned int)(2 * pair) + 1;   // paths: 2*pair and 2*pair+1
    unsigned int g = idx ^ (idx >> 1);
    unsigned int shift = get_shift(2 * pair);          // shift for first path

    // Generate normals for the first path
    for (int i = 0; i < params.N; i++) {
        unsigned int x = 0;
        for (int b = 0; b < 31; b++) {
            if (g & (1u << b))
                x ^= sobol_dirs[i * 31 + b];
        }
        x ^= shift;
        double u = ((double)x + 0.5) * 2.3283064365386963e-10;
        z[i] = inverse_normal(u);
        if (!isfinite(z[i])) z[i] = 0.0;
    }
    brownian_bridge(z, params.N, params.dt);
    // Save a copy of the increments for the second (antithetic) path
    double z_anti[128];
    for (int i = 0; i < params.N; i++) z_anti[i] = -z[i];

    // Compute payoffs for first path
    double logS = log(params.S0);
    double drift = (params.r - 0.5 * params.sigma * params.sigma) * params.dt;
    double vol = params.sigma * sqrt(params.dt);
    double sum_arith1 = 0.0, sum_geo1 = 0.0;
    for (int i = 0; i < params.N; i++) {
        logS += drift + vol * z[i];
        if (!isfinite(logS)) logS = 0.0;
        double S = exp(fmax(fmin(logS, 50.0), -50.0));
        sum_arith1 += S;
        sum_geo1 += logS;
    }
    double a1 = (sum_arith1 / params.N) - params.K;
    double g1 = exp(sum_geo1 / params.N) - params.K;

    // Compute payoffs for antithetic path
    logS = log(params.S0);
    double sum_arith2 = 0.0, sum_geo2 = 0.0;
    for (int i = 0; i < params.N; i++) {
        logS += drift + vol * z_anti[i];
        if (!isfinite(logS)) logS = 0.0;
        double S = exp(fmax(fmin(logS, 50.0), -50.0));
        sum_arith2 += S;
        sum_geo2 += logS;
    }
    double a2 = (sum_arith2 / params.N) - params.K;
    double g2 = exp(sum_geo2 / params.N) - params.K;

    // Average the two antithetic payoffs
    double a_avg = 0.5 * (fmax(a1, 0.0) + fmax(a2, 0.0));
    double g_avg = 0.5 * (fmax(g1, 0.0) + fmax(g2, 0.0));

    arith[pair] = isfinite(a_avg) ? a_avg : 0.0;
    geo[pair]   = isfinite(g_avg) ? g_avg : 0.0;
}

// --- Host code ---

CudaQOMCE::CudaQOMCE(const AOP& params) : gpu_params_(params) {
    size_t num_elements = 512 * 31;
    if (d_sobol_ptr == nullptr) {
        std::vector<unsigned int> flat_sobol(num_elements);
        for (int i = 0; i < 512; i++) {
            for (int j = 0; j < 31; j++) {
                flat_sobol[i * 31 + j] = (unsigned int)sobol_data::kDirectionNumbers[i][j];
            }
        }
        cudaMalloc(&d_sobol_ptr, num_elements * sizeof(unsigned int));
        cudaMemcpy(d_sobol_ptr, flat_sobol.data(), num_elements * sizeof(unsigned int), cudaMemcpyHostToDevice);
    }
}

double normal_cdf(double x) { return 0.5 * erfc(-x * 0.70710678118); }

// Undiscounted geometric Asian exact price
double analytic_geometric_asian(const urop::GPUParams& p) {
    double T = p.N * p.dt;
    double sig_hat = p.sigma * std::sqrt((p.N + 1.0) * (2.0 * p.N + 1.0) / (6.0 * p.N * p.N));
    double mu_hat = (p.r - 0.5 * p.sigma * p.sigma) * (p.N + 1.0) / (2.0 * p.N) + 0.5 * sig_hat * sig_hat;
    double d1 = (std::log(p.S0 / p.K) + (mu_hat + 0.5 * sig_hat * sig_hat) * T) / (sig_hat * std::sqrt(T));
    double d2 = d1 - sig_hat * std::sqrt(T);
    double price = p.S0 * std::exp(mu_hat * T) * normal_cdf(d1) - p.K * normal_cdf(d2); // undiscounted
    return isfinite(price) ? price : 0.0;
}

MCResult CudaQOMCE::run() {
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);
    // M is now number of pairs (total original paths = 2 * M)
    long long pairs = gpu_params_.M;   // assume M is number of pairs
    double *d_arith, *d_geo;

    cudaMalloc(&d_arith, sizeof(double) * pairs);
    cudaMalloc(&d_geo, sizeof(double) * pairs);
    cudaMemset(d_arith, 0, sizeof(double) * pairs);
    cudaMemset(d_geo, 0, sizeof(double) * pairs);

    int threads = 64;
    int blocks = (int)((pairs + threads - 1) / threads);

    asian_qmc_kernel<<<blocks, threads>>>(gpu_params_, d_arith, d_geo, d_sobol_ptr);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return {0.0, 0.0};
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        return {0.0, 0.0};
    }

    std::vector<double> h_arith(pairs), h_geo(pairs);
    cudaMemcpy(h_arith.data(), d_arith, sizeof(double) * pairs, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_geo.data(), d_geo, sizeof(double) * pairs, cudaMemcpyDeviceToHost);

    cudaFree(d_arith);
    cudaFree(d_geo);

    // ---- Debug: print first 10 averaged payoffs ----
    printf("First 10 averaged arith payoffs (per pair):\n");
    for (int i = 0; i < 10 && i < pairs; ++i)
        printf("  %d: %f\n", i, h_arith[i]);
    printf("First 10 averaged geo payoffs:\n");
    for (int i = 0; i < 10 && i < pairs; ++i)
        printf("  %d: %f\n", i, h_geo[i]);

    long long valid = 0;
    for (long long i = 0; i < pairs; ++i) {
        if (std::isfinite(h_arith[i]) && std::isfinite(h_geo[i]))
            ++valid;
    }
    if (valid == 0) {
        printf("Warning: No valid paths generated. Check CUDA kernel.\n");
        return {0.0, 0.0};
    }

    BiRunStats cv;
    for (long long i = 0; i < pairs; ++i) {
        if (std::isfinite(h_arith[i]) && std::isfinite(h_geo[i]))
            cv.update(h_arith[i], h_geo[i]);
    }

    double beta = 0.0;
    if (cv.get_count() > 1) {
        beta = cv.beta();
        if (!std::isfinite(beta)) beta = 0.0;
    }

    double geo_exact = analytic_geometric_asian(gpu_params_); // undiscounted
    printf("geo_exact (undiscounted) = %f\n", geo_exact);

    RunStats final_stats;
    for (long long i = 0; i < pairs; ++i) {
        double val = h_arith[i];
        if (std::isfinite(h_arith[i]) && std::isfinite(h_geo[i])) {
            val = h_arith[i] - beta * (h_geo[i] - geo_exact);
        }
        if (!std::isfinite(val))
            val = std::isfinite(h_arith[i]) ? h_arith[i] : 0.0;
        final_stats.update(val);
    }

    return {
        final_stats.get_mean() * gpu_params_.discount,
        final_stats.get_std_error() * gpu_params_.discount
    };
}

} // namespace urop
