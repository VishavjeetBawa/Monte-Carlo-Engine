#pragma once

#include <memory>
#include "OptionParams.hpp"
#include "RunStats.hpp"
#include "Payoff.hpp"
#include "RNG.hpp"

namespace urop{

struct MCResult{
    double price;
    double std_error;
};


//crude i.e. without antithetic
class CrudeMCE{
public:

    CrudeMCE(const AOP params , std::unique_ptr<Payoff> payoff , std::unique_ptr<AbstractRNG> rng): 
        params_(params) , payoff_(std::move(payoff)) , rng_(std::move(rng)){}
    MCResult run();

private:
    const AOP params_;
    std::unique_ptr<Payoff> payoff_;
    std::unique_ptr<AbstractRNG> rng_;
    double calculate_path_payoff(const std::vector<double>& deviates , std::vector<double>& path_prices)const;
}; 



//Antithetic variates version
class AVMCE{
public:

    AVMCE(const AOP params , std::unique_ptr<Payoff> payoff , std::unique_ptr<AbstractRNG> rng): 
        params_(params) , payoff_(std::move(payoff)) , rng_(std::move(rng)){}
    MCResult run();

private:
    const AOP params_;
    std::unique_ptr<Payoff> payoff_;
    std::unique_ptr<AbstractRNG> rng_;
    double calculate_path_payoff(const std::vector<double>& deviates , std::vector<double>& path_prices)const;

};


//Control variate + antithetic i.e. most optimised version of mc

class OMCE{
public:
    OMCE(const AOP params , std::unique_ptr<Payoff> arith_payoff , std::unique_ptr<Payoff> geo_payoff , std::unique_ptr<AbstractRNG> rng , double geo_exact):
        params_(params) , arith_payoff_(std::move(arith_payoff)) , geo_payoff_(std::move(geo_payoff)) , rng_(std::move(rng)) , geo_exact_(geo_exact){}

    MCResult run();
private:
    const AOP params_;
    std::unique_ptr<Payoff> arith_payoff_;
    std::unique_ptr<Payoff> geo_payoff_;
    std::unique_ptr<AbstractRNG> rng_;
    const double geo_exact_; //Exact geometric value of this 
    double calculate_path_payoff(const std::vector<double>& deviates , std::vector<double>& path_prices , Payoff& payoff)const;
};

};
