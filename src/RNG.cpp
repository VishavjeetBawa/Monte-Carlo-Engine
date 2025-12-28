#include "RNG.hpp"
#include <vector>
#include <algorithm>

namespace urop{

void MtRand::generate_deviates(long long count , std::vector<double>& deviates){
    deviates.resize(count);
    std::generate(deviates.begin() , deviates.end() , [&](){
        return distribution_(generator_);
    });

}

}
