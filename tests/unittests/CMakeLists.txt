add_executable(embedding
    test_input_embedding.cu
)
target_link_libraries(
    embedding PUBLIC
    -lcudart
    -lcudadevrt
    embeddingFunctor)



add_executable(rmsnorm
    test_rmsnorm.cu
)
target_link_libraries(
    rmsnorm PUBLIC
    -lcudart
    -lcudadevrt
    rmsNormFunctor)

add_executable(testlinear
    test_linear.cu
)
target_link_libraries(  # Libs for test_build_casual_mask
    testlinear PUBLIC
    -lcudart
    -lcudadevrt
    linear) 
