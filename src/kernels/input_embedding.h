#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "src/weights/llama/embedding_weights.h"
#include "src/utils/tensor.h"


template<typename T>
void launchInputEmbedding(TensorWrapper<int>* input_ids,    // INT [token num]
                          TensorWrapper<T>* output,       // FP32 [token num, hidden_size] = [token num, 4096]
                          EmbeddingWeight<T>* embed_table// FP32 [vocal_size, hidden_size]
                          );