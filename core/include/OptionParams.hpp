#pragma once

#include <cmath>
#include <stdexcept>

namespace urop{

struct AOP{//asian option parameters

// Financial necessity
    const double S0_; 
    const double K_;  
    const double T_;  
    const double R_;  
    const double sigma_;  
    
// Overhead
    const int N_; //number of time steps per path
    const long long int M_; //numbere of paths
    const double dT_; // this is the (T_/N_) which tells us the even change in time to be taken
    const double dF_; //discount factor
    

    AOP(double s0 , double k , double t , double r  ,double sigma , int n , long long int m):S0_(s0) , K_(k) , T_(t) , R_(r),
        sigma_(sigma) , N_(n) , M_(m) , dT_(T_/N_) , dF_(std::exp(-R_*T_)){
        
        if(T_<=0.0 || sigma_<=0.0 || N_<=0 || M_<=0){
            throw std::invalid_argument("Check the values initialised for T_ , sigma_ , N_ and M_ under AOP");
        }
    }
};

};
