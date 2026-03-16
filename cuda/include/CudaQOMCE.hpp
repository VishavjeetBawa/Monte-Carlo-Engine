#pragma once
#include "GPUParams.hpp"
#include "MCE.hpp"

namespace urop {

class CudaQOMCE {
public:
    explicit CudaQOMCE(const AOP& params);
    MCResult run();

private:
    GPUParams gpu_params_;
    void adjust_stack_limit();
};

}
