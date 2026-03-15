#pragma once

#include "GPUParams.hpp"
#include "OptionParams.hpp"
#include "MCE.hpp"

namespace urop {

class CudaQOMCE {

public:

    explicit CudaQOMCE(const AOP& params);

    MCResult run();

private:

    GPUParams gpu_params_;

};

}
