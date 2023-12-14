/*!
@file cuda_utils.cuh

@brief Header file containing function declarations of cuda utility functions

### Author(s)
- Created by Aidan Vickars on Dec 13, 2023

*/

#ifndef BARCO_CUDA_UTILS_CUH
#define BARCO_CUDA_UTILS_CUH

#include<cuda_runtime.h>

/**
 * @brief Checks for a cuda error
 * @param cuda_status
 * @param message
 */
void check_cuda_error(cudaError_t cuda_status, const char* message = nullptr);

/**
 * @brief Allocates device memory
 */
void allocate_device_memory(void*&, size_t);

/**
 * @brief Allocates 2d device memory
 */
void allocate_2d_device_memory(void*&, size_t&, size_t, size_t);

/**
 * @brief Deallocates device memory
 */
void deallocate_device_memory(void*t);

/**
 * @brief Copies device memory from a 2d array to another 2d array
 */
void copy_2d(void*, size_t, void*, size_t, size_t, size_t, cudaMemcpyKind);

/**
 * @brief Copies device memory
 */
void copy(void*, void*, size_t, cudaMemcpyKind);

#endif //BARCO_CUDA_UTILS_CUH
