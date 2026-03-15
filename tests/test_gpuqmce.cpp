#include "OptionParams.hpp"
#include "CudaQOMCE.hpp"
#include "Timer.hpp"
#include <iomanip>
#include <iostream>

int main()
{
    // Parameters (Standard Asian Option Test Case)
    urop::AOP params(100.0,   // S0
                     100.0,   // K
                     1.0,     // T
                     0.05,    // r
                     0.2,     // sigma
                     100,     // N (time steps)
                     100000); // M (total paths)

    // Initialise the GPU Engine
    // Note: Your CudaQOMCE handles its own Sobol sequences and 
    // Arithmetic Average payoff logic within the CUDA kernel.
    urop::CudaQOMCE engine(params);

    urop::Timer timer;

    // Run the GPU Simulation
    timer.start();
    urop::MCResult result = engine.run();
    timer.stop();

    // Output Results in your specific format
    std::cout << "COQMCE Price: " << std::fixed << std::setprecision(6)
              << result.price << "\n";
    std::cout << "Std Error: " << result.std_error << "\n";

    // Benchmark Report
    // Label updated to reflect GPU execution
    timer.print_report("Concurrent QMC (Sobol + BB + Shift + AV + CV + GPU)",
                       params.M_);

    return 0;
}

