#include "src/kernels/act_kernel.h"

#include "src/utils/cuda_debug_utils.cuh"
#include "src/utils/macro.h"

template<typename T>
__device__ __forceinline__ T silu(const T& in) {
  float x = float(in);
  return (T)(x / (1.0f + expf(-x)));
}

template<>
__device__ __forceinline__ half2 silu<half2>(const half2& in) {
  float x = __half2float(in.x);
  float y = __half2float(in.y);
  x = x / (1.0f + expf(-x));
  y = y / (1.0f + expf(-y));

  return make_half2(__float2half(x), __float2half(y));
}



template<typename T>
__global__ void silu_and_mul_kernel(T* out,               // [bs, intermedia size]
                                    const T* input,       // [bs, 2, intermedia size]
                                    const int intermedia_size) 
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  for(; tid < intermedia_size; tid += blockDim.x)
  {
    const T x = input[bid * 2 * intermedia_size + tid];
    const T y = input[bid * 2 * intermedia_size + tid + intermedia_size];
    out[bid * intermedia_size + tid] = silu<T>(x) * y;
  }
}


template<>
__global__ void silu_and_mul_kernel<half>(half* out,               // [bs, intermedia size]
                                          const half* input,       // [bs, 2, intermedia size]
                                          const int intermedia_size)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  using Vec_t = typename Vec<half>::Type;
  int vec_size = Vec<half>::size;

  for(tid = vec_size * tid; tid < intermedia_size; tid += vec_size * blockDim.x)
  {
    const Vec_t x = *reinterpret_cast<Vec_t*>(const_cast<half*>(&input[bid * 2 * intermedia_size + tid]));
    const Vec_t y = *reinterpret_cast<Vec_t*>(const_cast<half*>(&input[bid * 2 * intermedia_size + tid + intermedia_size]));
    
    *reinterpret_cast<Vec_t*>(&out[bid * intermedia_size + tid]) = __hmul2(silu<Vec_t>(x), y);
  }
}

template<typename T>
void launchAct(TensorWrapper<T>* input, TensorWrapper<T>* out) {
    int batch_size = input->shape[0];
    LLM_CHECK(input->shape[1] == 2);
    int intermedia_size = input->shape[2];
    dim3 grid(batch_size);
    dim3 block(256);
    silu_and_mul_kernel<T><<<grid, block>>>(out->data, input->data, intermedia_size);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(out->data);
#else
#endif
}
// We must instancite the template, if not, will report linking issue
template void launchAct(TensorWrapper<float>* input, TensorWrapper<float>* output);
template void launchAct(TensorWrapper<half>* input, TensorWrapper<half>* output);
