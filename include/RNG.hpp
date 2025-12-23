#pragma once

#include <vector>
#include <random>
#include <chrono>

namespace urop{

class AbstractRNG{
public:
    virtual ~AbstractRNG() = default ;
    virtual std::vector<double> generate_deviates(long long count , std::vector<double>& deviates) const = 0 ;
};

class MtRand final : public AbstractRNG{
public:
    MtRand():generator_(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count())){}
    std::vector<double> generate_deviates(long long count , std::vector<double>& deviates) const override;

private:
    mutable std::mt19937 generator_;
    mutable std::normal_distribution<double> distribution_{0.0 , 1.0};
};

}
