#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "src/utils/tensor.h"
#include "src/weights/llama/rmsnorm_weights.h"
#include "src/utils/vectorize_utils.h"

template<typename T>
void launchRMSNorm(TensorWrapper<T>* activation,
                    TensorWrapper<T>* residual,
                    RMSNormWeight<T>& weight,
                    float eps,
                    bool is_last = false);

