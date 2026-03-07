
#include <vector>
#include "RNG.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <boost/math/distributions/normal.hpp>

namespace urop{

void MtRand::generate_deviates(long long count , std::vector<double>& deviates){
    deviates.resize(count);
    std::generate(deviates.begin() , deviates.end() , [&](){
        return distribution_(generator_);
    });

}

static inline double fast_normal_icdf(double u)
{
    // Peter J. Acklam approximation (very accurate)
    static const double a1 = -3.969683028665376e+01;
    static const double a2 =  2.209460984245205e+02;
    static const double a3 = -2.759285104469687e+02;
    static const double a4 =  1.383577518672690e+02;
    static const double a5 = -3.066479806614716e+01;
    static const double a6 =  2.506628277459239e+00;

    static const double b1 = -5.447609879822406e+01;
    static const double b2 =  1.615858368580409e+02;
    static const double b3 = -1.556989798598866e+02;
    static const double b4 =  6.680131188771972e+01;
    static const double b5 = -1.328068155288572e+01;

    static const double c1 = -7.784894002430293e-03;
    static const double c2 = -3.223964580411365e-01;
    static const double c3 = -2.400758277161838e+00;
    static const double c4 = -2.549732539343734e+00;
    static const double c5 =  4.374664141464968e+00;
    static const double c6 =  2.938163982698783e+00;

    static const double d1 =  7.784695709041462e-03;
    static const double d2 =  3.224671290700398e-01;
    static const double d3 =  2.445134137142996e+00;
    static const double d4 =  3.754408661907416e+00;

    double q, r;

    if (u < 0.02425) {
        q = std::sqrt(-2 * std::log(u));
        return (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
               ((((d1*q + d2)*q + d3)*q + d4)*q + 1);
    }
    else if (u > 1 - 0.02425) {
        q = std::sqrt(-2 * std::log(1 - u));
        return -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
                 ((((d1*q + d2)*q + d3)*q + d4)*q + 1);
    }
    else {
        q = u - 0.5;
        r = q * q;
        return (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q /
               (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1);
    }
}

std::vector<double> Sobol::next_point()
{
    if (current_index_ == 0)
        current_index_ = 1;

    uint32_t changedBit = rightmost_zero_bit(current_index_ - 1);
    if (changedBit >= sobol_data::kMaxSobolDigits)
        throw std::runtime_error("Sobol bit index exceeded Joe-Kuo limit");

    std::vector<double> point(dimensions_);
    for (uint32_t dim = 0; dim < dimensions_; ++dim)
    {
        uint32_t v = sobol_data::kDirectionNumbers[dim + 1][changedBit];
        current_point_[dim] ^= (v << (31 - changedBit));

        point[dim] = static_cast<double>(current_point_[dim]) / 4294967296.0; // 2^32
    }

    ++current_index_;
    return point;
}


uint32_t Sobol::rightmost_zero_bit(uint64_t n) const{
    uint32_t c = 0;
    while(n & 1ULL){
        n>>=1;
        ++c;
    }
    return c;
} 

static void brownian_bridge(std::vector<double>& Z, double T)
{
    const int N = static_cast<int>(Z.size());

    std::vector<double> W(N + 1, 0.0);   // Brownian values
    std::vector<int> left(N + 1), right(N + 1);

    // Time grid
    std::vector<double> t(N + 1);
    for (int i = 0; i <= N; ++i)
        t[i] = T * i / N;

    // Initial: endpoints
    W[0] = 0.0;
    W[N] = std::sqrt(T) * Z[0];

    left[0]  = 0;
    right[0] = N;

    int used = 1;

    // Bridge construction
    while (used < N)
    {
        for (int k = 0; k < used; ++k)
        {
            int l = left[k];
            int r = right[k];
            int m = (l + r) / 2;

            if (m == l || m == r) continue;

            double tl = t[l];
            double tr = t[r];
            double tm = t[m];

            double mean = ((tr - tm) * W[l] + (tm - tl) * W[r]) / (tr - tl);
            double var  = (tm - tl) * (tr - tm) / (tr - tl);

            W[m] = mean + std::sqrt(var) * Z[used++];

            left[k]  = l;
            right[k] = m;

            left[used - 1]  = m;
            right[used - 1] = r;

            if (used >= N) break;
        }
    }

    // Convert Brownian path to increments
    for (int i = 0; i < N; ++i)
        Z[i] = (W[i + 1] - W[i]) / std::sqrt(t[i + 1] - t[i]);
}


Sobol::Sobol(uint32_t dimensions, double T)
    : dimensions_(dimensions), T_(T), current_index_(0)
{
    if (dimensions == 0)
        throw std::invalid_argument("dim can't be 0");
    if (dimensions > 21201)
        throw std::invalid_argument("dim too high");

    current_point_.resize(dimensions_, 0);
    shift_.resize(dimensions_, 0.0);
}



void Sobol::generate_deviates(long long count, std::vector<double>& deviates)
{
    deviates.resize(count);

    // one Sobol point gives N uniforms
    auto uvec = next_point();

    for (int i = 0; i < dimensions_; ++i)
    {
        // digital shift
        double u = std::fmod(uvec[i] + shift_[i], 1.0);

        if (u <= 0.0) u = 1e-15;
        if (u >= 1.0) u = 1.0 - 1e-15;

        deviates[i] = fast_normal_icdf(u);
    }

    // Brownian bridge reorder
    brownian_bridge(deviates, T_);
}


void Sobol::randomize_shift()
{
    static thread_local std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<double> U(0.0, 1.0);

    for (uint32_t i = 0; i < dimensions_; ++i)
        shift_[i] = U(gen);
}

void Sobol::reset()
{
    current_index_ = 0;
    std::fill(current_point_.begin(), current_point_.end(), 0);
}


}
