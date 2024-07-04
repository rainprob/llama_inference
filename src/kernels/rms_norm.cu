#include "src/kernels/rms_norm.h"



template <typename T>
__device__ T warpReduce(T val)
{
    for(int i = 32 / 2; i > 0; i >>= 1)
    {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}

template <typename T>
__device__ T blockReduce(T val)
{
    int tid = threadIdx.x;
    int wid = tid / 32;
    int laneid = tid % 32;
    int warpnum = (blockDim.x + 32 - 1) / 32;

    __shared__ T warpsum[32];

    val = warpReduce<T>(val);
    
    if(laneid == 0) warpsum[wid] = val;
    __syncthreads();
    
    T sum = tid < warpnum ? warpsum[wid] : (T)0;
    sum = warpReduce<T>(sum);

    return sum;
}

template <typename T>
__global__ void rmsNormFunctor(T* activation,
                                T* residual,
                                T* weight,
                                float eps,
                                int hidden_state
                                )
{
    using Vec_t = typename Vec<T>::Type;
    int vec_size = Vec<T>::size;

    Vec_t* data = reinterpret_cast<Vec_t*>(activation + blockIdx.x * hidden_state);
    Vec_t* rsd = reinterpret_cast<Vec_t*>(residual + blockIdx.x * hidden_state);
    float thread_sum = 0.0f;

    for(int i = threadIdx.x; i < hidden_state / vec_size; i += blockDim.x)
    {
        
        Vec_t vec = data[i];
        rsd[i] = vec;
        thread_sum += vec.x * vec.x;
        thread_sum += vec.y * vec.y;
        thread_sum += vec.z * vec.z;
        thread_sum += vec.w * vec.w;       
    }

    thread_sum = blockReduce<T>(thread_sum);

    __shared__ float inv_mean;
    
    if(threadIdx.x == 0)    inv_mean = rsqrtf(thread_sum / hidden_state + eps);
    __syncthreads();

    Vec_t* w = reinterpret_cast<Vec_t*>(weight);
   
    for(int i = threadIdx.x; i < hidden_state / vec_size; i += blockDim.x)
    {
        data[i].x = data[i].x * inv_mean * w[i].x;
        data[i].y = data[i].y * inv_mean * w[i].y;
        data[i].z = data[i].z * inv_mean * w[i].z;
        data[i].w = data[i].w * inv_mean * w[i].w;
    }
}

template <>
__global__ void rmsNormFunctor(half* activation,
                                half* residual,
                                half* weight,
                                float eps,
                                int hidden_state
                                )
{
    using Vec_t = typename Vec<half>::Type;
    int vec_size = Vec<half>::size;

    Vec_t* data = reinterpret_cast<Vec_t*>(activation + blockIdx.x * hidden_state);
    Vec_t* rsd = reinterpret_cast<Vec_t*>(residual + blockIdx.x * hidden_state);
    float thread_sum = 0.0f;

    for(int i = threadIdx.x; i < hidden_state / vec_size; i += blockDim.x)
    {
        
        Vec_t vec = data[i];
        rsd[i] = vec;
        thread_sum += __half2float(vec.x) * __half2float(vec.x);
        thread_sum += __half2float(vec.y) * __half2float(vec.y);
    }

    thread_sum = blockReduce<half>(thread_sum);

    __shared__ float inv_mean;
    
    if(threadIdx.x == 0)    inv_mean = rsqrtf(thread_sum / hidden_state + eps);
    __syncthreads();

    Vec_t* w = reinterpret_cast<Vec_t*>(weight);
   
    for(int i = threadIdx.x; i < hidden_state / vec_size; i += blockDim.x)
    { 
        data[i].x = __float2half(__half2float(data[i].x) * inv_mean) * w[i].x;
        data[i].y = __float2half(__half2float(data[i].y) * inv_mean) * w[i].y;
    }
}


template<typename T>
void launchRMSNorm(TensorWrapper<T>* activation,
                    TensorWrapper<T>* residual,
                    RMSNormWeight<T>& weight,
                    float eps,
                    bool is_last)
{
    int hidden_state = activation->shape[1];

    int grid_size = activation->shape[0];
    int block_size = hidden_state / Vec<T>::size;

    rmsNormFunctor<T><<<grid_size, block_size>>>(activation->data,
                                                residual->data,
                                                weight.gamma,
                                                eps,
                                                hidden_state);
    
}

template
void launchRMSNorm(TensorWrapper<float>* activation,
                    TensorWrapper<float>* residual,
                    RMSNormWeight<float>& weight,
                    float eps,
                    bool is_last);

template
void launchRMSNorm(TensorWrapper<half>* activation,
                    TensorWrapper<half>* residual,
                    RMSNormWeight<half>& weight,
                    float eps,
                    bool is_last);