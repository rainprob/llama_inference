#include "src/kernels/input_embedding.h"


template <typename T>
__global__ void embeddingFunctor(const int* input_ids,
                                    T* output,
                                    const T* emebed_table,
                                    const int max_seq_len,
                                    const int hidden_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = idx; i < max_seq_len * hidden_size; i += stride)
    {
        int token_idx = idx / hidden_size;
        int token = input_ids[tokenIdx];
        int embed_idx = token_idx * hidden_size + (idx % hidden_size)
        output[idx] = emebed_table[embed_idx]
    }
    
}


// can not pass const input, cause the data is not const.
template <typename T>
void launchInputEmbedding(TensorWrapper<int>* input_ids,
                            TensorWrapper<T>* output,
                            TensorWrapper<T>* emebed_table)
{
    const int blockSize = 256;
    const int gridSize = 2048;
    const int hidden_size = emebed_table->shape[1];
    const int max_seq_len = output->shape[0];

    LLM_CHECK_WITH_INFO(max_seq_len == input_ids->shape[0], "input ids 1st shape should equal to 1st shape of output");

    embeddingFunctor<<<gridSize, blockSize>>>(input_ids->data,
                                                output->data,
                                                emebed_table->data,
                                                max_seq_len,
                                                hidden_size)

}




template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                                   TensorWrapper<float>* output,       
                                   EmbeddingWeight<float>* embed_table);
template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                                   TensorWrapper<half>* output,       
                                   EmbeddingWeight<half>* embed_table);
