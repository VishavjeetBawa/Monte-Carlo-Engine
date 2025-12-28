#pragma once

#include <vector>
#include <random>
#include <chrono>

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

}
