#include "MonteCarlo.hpp"
#include <iomanip>

int main(){

    urop::AOP params(100.0, 100.0, 1.0, 0.05, 0.2, 100, 100000); // s0 , k , t , r , sigma , n , m
    auto geo_payoff = std::make_unique<urop::GeometricAsianPayoff>(params.K_);
    auto arith_payoff = std::make_unique<urop::AsianCallPayoff>(params.K_);
    auto rng = std::make_unique<urop::MtRand>();

    urop::geo_pricer geo_exact(params);
    double known = geo_exact.price();

    urop::OMCE engine(params , std::move(arith_payoff) ,std::move(geo_payoff) , std::move(rng) , known);

    urop::Timer timer;
    
    timer.start();
    urop::MCResult result = engine.run();
    timer.stop();

    std::cout << "MC Price: " << std::fixed << std::setprecision(6) << result.price << "\n";
    std::cout << "Std Error: " << result.std_error << "\n";

    timer.print_report("Optimised MC (AV+CV)", params.M_);
    
    return 0;
}
