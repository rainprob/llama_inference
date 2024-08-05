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

add_executable(paddingoffset
    test_cal_paddingoffset.cu
)
target_link_libraries(
  paddingoffset PUBLIC
    -lcudart
    -lcudadevrt
    cal_paddingoffset) 

add_executable(causalmask
    test_casual_mask.cu
)
target_link_libraries(  # Libs for test_build_casual_mask
   causalmask PUBLIC
    -lcudart
    -lcudadevrt
    build_casual_mask) 

add_executable(testlinear
    test_linear.cu
)
target_link_libraries(  # Libs for test_build_casual_mask
    testlinear PUBLIC
    -lcudart
    -lcudadevrt
    linear) 

add_executable(test_fused_trans_remv_pad
    test_fused_trans_remv_pad.cu
)
target_link_libraries(  
    test_fused_trans_remv_pad PUBLIC
    -lcudart
    -lcudadevrt
    fused_transpose_and_remv_pad) 


add_executable(test_fused_addresidual_norm
    test_fused_addresidual_norm.cu
)
target_link_libraries(  
    test_fused_addresidual_norm PUBLIC
    -lcudart
    -lcudadevrt
    fused_addresidual_norm) 

add_executable(test_act
    test_act.cu
)
target_link_libraries(  
    test_act PUBLIC
    -lcudart
    -lcudadevrt
    act) 