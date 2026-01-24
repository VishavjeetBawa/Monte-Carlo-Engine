#pragma once

//RNG.hpp

#include "joe_kuo_sobol_data.hpp"

#include <vector>
#include <random>
#include <chrono>
#include <stdexcept>

namespace urop{

class AbstractRNG{
public:
    virtual ~AbstractRNG() = default ;
    virtual void generate_deviates(long long count , std::vector<double>& deviates)  = 0 ;
};

class MtRand final : public AbstractRNG{
public:
    MtRand():generator_(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count())){}
    void generate_deviates(long long count , std::vector<double>& deviates) override;

private:
    std::mt19937 generator_;
    std::normal_distribution<double> distribution_{0.0 , 1.0};
};



/**
****************************************
Following is the sobol implementation:-
****************************************
**/
class Sobol final : public AbstractRNG {
public:
    Sobol(uint32_t dimensions, double T);

    void generate_deviates(long long count, std::vector<double>& deviates) override;

    void reset();
    void randomize_shift();

private:
    std::vector<double> next_point();
    uint32_t rightmost_zero_bit(uint64_t n) const;

private:
    uint32_t dimensions_;
    double T_;

    uint64_t current_index_;
    std::vector<uint32_t> current_point_;

    std::vector<double> shift_;   // digital shift
};


}
