/*!
@file cuda_utils.cu

@brief Source file containing function definitions of cuda utility functions

### Author(s)
- Created by Aidan Vickars on Dec 13, 2023

*/

#include "cuda_utils.cuh"
#include <iostream>

/**
 * Function that checks if a cuda error was received
 * @param cuda_status Status of the cuda function
 * @param message Message to print if an error was received
 */
void check_cuda_error(cudaError_t cuda_status, const char* message) {
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(cuda_status);
        if (message) {
            std::cerr << " (" << message << ")";
        }
        std::cerr << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * Allocates device memory
 * @param device_ptr Pointer to allocate memory to
 * @param size The size of the memory allocation (in bytes)
 */
void allocate_device_memory(void*& device_ptr, size_t size) {
    check_cuda_error(cudaMalloc((void **) &device_ptr, size));
}

/**
 * Allocates 2d device memory
 * @param device_ptr Pointer to allocate memory to
 * @param device_pitch The pitch of the allocated memory
 * @param width_bytes The minimum width of the memory allocation (in bytes)
 * @param height The height of the memory allocation
 */
void allocate_2d_device_memory(void*& device_ptr, size_t& device_pitch, size_t width_bytes, size_t height) {
    check_cuda_error(cudaMallocPitch((void **) &device_ptr, &device_pitch, width_bytes, height));
}

/**
 * Deallocates device memory
 * @param device_ptr Pointer containing memory to deallocate
 */
void deallocate_device_memory(void* device_ptr) {
    check_cuda_error(cudaFree((void *) device_ptr));
}

/**
 * Copies memory from a 2D array to another 2D array.  The copy need not be from device to device.  Can be from
 * host to device, device to host, etc.
 * @param dst_ptr The pointer of the destination memory
 * @param dst_pitch The point of the destination memory
 * @param src_ptr The pointer of the source memory
 * @param src_pitch The pitch of the source memory
 * @param width_bytes The width of the copy (in bytes)
 * @param height The height of the copy
 * @param kind The type of copy. See here for possible types: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g18fa99055ee694244a270e4d5101e95b
 */
void copy_2d(void* dst_ptr, size_t dst_pitch, void* src_ptr, size_t src_pitch, size_t width_bytes, size_t height, cudaMemcpyKind kind) {
    check_cuda_error(cudaMemcpy2D((void *) dst_ptr, dst_pitch, (void *) src_ptr, src_pitch, width_bytes, height, kind));
}

/**
 * Copies memory from an array to another array.  The copy need not be from device to device.  Can be from
 * host to device, device to host, etc.
 * @param dst The pointer of the destination memory
 * @param src The pointer of the source memory
 * @param size The size (in bytes) of the copy
 * @param kind The type of copy. See here for possible types: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g18fa99055ee694244a270e4d5101e95b
 */
void copy(void* dst, void* src, size_t size, cudaMemcpyKind kind) {
    check_cuda_error(cudaMemcpy(dst, src, size, kind));
}