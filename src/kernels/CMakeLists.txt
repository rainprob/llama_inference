add_library(embeddingFunctor STATIC input_embedding.cu)
set_property(TARGET embeddingFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET embeddingFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET embeddingFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)


add_library(rmsNormFunctor STATIC rms_norm.cu)
set_property(TARGET rmsNormFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET rmsNormFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET rmsNormFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(cublasWrapper STATIC cublas_utils.cc)
set_property(TARGET cublasWrapper PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasWrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(linear STATIC linear.cu)
set_property(TARGET linear PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET linear PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET linear PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(linear PUBLIC -lcudart -lcublas cublasWrapper)