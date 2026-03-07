#include "MonteCarlo.hpp"
#include <iomanip>

int main()
{
    // Parameters (same as OMCE)
    urop::AOP params(100.0,   // S0
                     100.0,   // K
                     1.0,     // T
                     0.05,    // r
                     0.2,     // sigma
                     100,     // N  (time steps)
                     100000); // M  (total paths)

    // Payoffs
    auto geo_payoff   = std::make_unique<urop::GeometricAsianPayoff>(params.K_);
    auto arith_payoff = std::make_unique<urop::AsianCallPayoff>(params.K_);

    // Sobol RNG prototype (important: this is cloned per thread internally)
    auto rng = std::make_unique<urop::Sobol>(params.N_, params.T_);

    // Exact geometric price for control variate
    urop::geo_pricer geo_exact(params);
    double known = geo_exact.price();

    // Concurrent engine (no batching needed here — threads are the batching)
    urop::COQMCE engine(params,
                        std::move(arith_payoff),
                        std::move(geo_payoff),
                        std::move(rng),
                        known);

    urop::Timer timer;

    timer.start();
    urop::MCResult result = engine.run();
    timer.stop();

    std::cout << "COQMCE Price: " << std::fixed << std::setprecision(6)
              << result.price << "\n";
    std::cout << "Std Error: " << result.std_error << "\n";

    timer.print_report("Concurrent QMC (Sobol + BB + Shift + AV + CV + Threads)",
                       params.M_);

    return 0;
}

