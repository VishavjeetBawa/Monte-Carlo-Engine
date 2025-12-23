#include "RNG.hpp"
#include <vector>
#include <algorithm>

namespace urop{

std::vector<double> MtRand::generate_deviates(long long count , std::vector<double>& deviates) const{
    deviates.resize(count);
    std::generate(deviates.begin() , deviates.end() , [&gen = generator_ , &dist = distribution_](){
        return dist(gen);
    });

    return deviates;
}

}
