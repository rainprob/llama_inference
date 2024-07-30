#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

enum class WeightType {
    FP32_W,
    FP16_W,
    INT8_W,
    UNSUPPORTED_W
};


template <typename T>
inline WeightType getWeightType()
{
    if (std::is_same<T, float>::value == 1 || std::is_same<T, const float>::value == 1) return WeightType::FP32_W;
    else if (std::is_same<T, half>::value == 1 || std::is_same<T, const half>::value == 1) return WeightType::FP16_W;
    else if (std::is_same<T, int8_t>::value == 1 || std::is_same<T, const int8_t>::value == 1) return WeightType::INT8_W;
    else return WeightType::UNSUPPORTED_W;
}


template <typename T>
struct BaseWeight
{
    WeightType type;
    std::vector<int> shape;
    T* data;
    T* bias;
};

