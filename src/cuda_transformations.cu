/*!
@file cuda_transformations.cu

@brief Source file containing function definitions of transformations using cuda

### Author(s)
- Created by Aidan Vickars on Dec 13, 2023

*/

#include "cuda_transformations.cuh"

#include "cuda_utils.cuh"

/**
 * The kernel of the tint transformation
 * @param input Pointer to the image in device memory
 * @param tints The tint to apply to the image.  Expects a 1*3 array
 * @param weight The weight to use when applying the pitch
 * @param height The height of the image
 * @param width The width of the image
 * @param pitch The pitch of the image in device memory
 */
__global__ void tint_kernel(uint8_t * input, const uint8_t* tints, float weight, size_t height, size_t width, size_t pitch) {

    // Computing target positions
    unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int z = threadIdx.z;

    // Out of bounds guard
    if ((x >= width) || (y >= height)) {
        return;
    }

    unsigned int pos = y * pitch / sizeof(uint8_t) + x * 3 + z;

    input[pos] = static_cast<unsigned char>((1.0 - weight) * input[pos] + weight * tints[z]);
}

/**
 * Function that applies a tint to the image using CUDA.
 * @param device_tints The tint to apply to the image.  Expects a 1*3 array representing the tints to apply to each channel
 * @param weight The weight to apply to the tint.  Expected to be in [0,1]
 * @param device_img_ptr Pointer to the image in device memory
 * @param device_img_pitch The pitch of the image in device memory
 * @param height Height of the image
 * @param width Width of the image
 */
void tint(const uint8_t* device_tints, float weight, uint8_t* device_img_ptr, size_t device_img_pitch, size_t height, size_t width) {
    // Creating threads
    dim3 blockDims(16, 16, 3);
    dim3 gridDims((width + blockDims.x - 1) / blockDims.x, (height + blockDims.y - 1) / blockDims.y, 1);

    // Executing kernal
    tint_kernel<<<gridDims, blockDims>>>(device_img_ptr, device_tints, weight, height, width, device_img_pitch);

    // Forcing completion
    cudaDeviceSynchronize(); // Synchronize to catch kernel launch errors
    check_cuda_error(cudaGetLastError(), "Tint kernel launch failed");

}

/**
 * Kernel of the normalization transformation
 * @param src_ptr Pointer to src image in device memory
 * @param src_pitch Pitch of the src image in device memory
 * @param dst_ptr Pointer to the allocated dst memory
 * @param dst_pitch Pitch of the allocated dst memory
 * @param height Height of the image
 * @param width Width of the image
 */
__global__ void normalize_kernel(const uint8_t* src_ptr, size_t src_pitch, float* dst_ptr, size_t dst_pitch, size_t height, size_t width) {

    // Computing target positions
    unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int z = threadIdx.z;

    // Out of bounds guard
    if ((x >= width) || (y >= height)) {
        return;
    }

    unsigned int src_pos = y * src_pitch / sizeof(uint8_t) + x * 3 + z;
    unsigned int dst_pos = y * dst_pitch  / sizeof(float) + x * 3 + z;

    dst_ptr[dst_pos] = static_cast<float>(src_ptr[src_pos] / 255.0);
}

/**
 * Function that normalizes an image by dividing by 255
 * @param src_ptr Pointer to src image in device memory.  Image is expected to have values in [0,255]
 * @param src_pitch Pitch of the src image in device memory
 * @param dst_ptr Pointer to the allocated dst memory
 * @param dst_pitch Pitch of the allocated dst memory
 * @param height Height of the image
 * @param width Width of the image
 */
void normalize(const uint8_t* src_ptr, size_t src_pitch, float* dst_ptr, size_t dst_pitch, size_t height, size_t width) {
    // Creating threads
    dim3 blockDims(16, 16, 3);
    dim3 gridDims((width + blockDims.x - 1) / blockDims.x, (height + blockDims.y - 1) / blockDims.y, 1);

    // Executing kernal
    normalize_kernel<<<gridDims, blockDims>>>(src_ptr, src_pitch, dst_ptr, dst_pitch, height, width);

    // Forcing completion
    cudaDeviceSynchronize(); // Synchronize to catch kernel launch errors
    check_cuda_error(cudaGetLastError(), "Normalize kernel launch failed");

}

/**
 * Kernel of the denormalization transformation
 * @param src_ptr Pointer to src image in device memory
 * @param src_pitch Pitch of the src image in device memory
 * @param dst_ptr Pointer to the allocated dst memory
 * @param dst_pitch Pitch of the allocated dst memory
 * @param height Height of the image
 * @param width Width of the image
 */
__global__ void denormalize_kernal(const float* src_ptr, size_t src_pitch, uint8_t* dst_ptr, size_t dst_pitch, size_t height, size_t width) {

    // Computing target positions
    unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int z = threadIdx.z;

    // Out of bounds guard
    if ((x >= width) || (y >= height)) {
        return;
    }

    unsigned int src_pos = y * src_pitch / sizeof(float) + x * 3 + z;
    unsigned int dst_pos = y * dst_pitch  / sizeof(uint8_t) + x * 3 + z;

    dst_ptr[dst_pos] = static_cast<uint8_t>(src_ptr[src_pos] * 255);
}

/**
 * Function that denormalizes an image by multiplying by 255
 * @param src_ptr Pointer to src image in device memory.  Image is expected to have values in [0,1]
 * @param src_pitch Pitch of the src image in device memory
 * @param dst_ptr Pointer to the allocated dst memory
 * @param dst_pitch Pitch of the allocated dst memory
 * @param height Height of the image
 * @param width Width of the image
 */
void denormalize(const float* src_ptr, size_t src_pitch, uint8_t* dst_ptr, size_t dst_pitch, size_t height, size_t width) {
    // Creating threads
    dim3 blockDims(16, 16, 3);
    dim3 gridDims((width + blockDims.x - 1) / blockDims.x, (height + blockDims.y - 1) / blockDims.y, 1);

    // Executing kernal
    denormalize_kernal<<<gridDims, blockDims>>>(src_ptr, src_pitch, dst_ptr, dst_pitch, height, width);

    // Forcing completion
    cudaDeviceSynchronize(); // Synchronize to catch kernel launch errors
    check_cuda_error(cudaGetLastError(), "Denormalize kernel launch failed");

}
/**
 * Kernel of the colour space transformation.  In this kernel we apply the transformation matrix on a single
 * (X/Y/Z) axis respectively in the XYZ CIE 1931 colour space. For instance, if we are applying the transformation to the x'th channel, we index
 * to the 1st row of the transformation matrix, and then perform a (1*3) times (3*1) matrix multiplication with
 * the (x,y) position rgb pixel value of the input image
 * @param m_matrix Pointer to the transformation matrix in device memory
 * @param src_ptr Pointer to the src image in device memory
 * @param src_pitch Pitch of the src image in device memory
 * @param dst_ptr Pointer to the dst image in device memory
 * @param dst_pitch Pitch of the dst image in device memory
 * @param height Height of the image
 * @param width Width of the image
 */
__global__ void colour_space_conversion_kernel(const float* m_matrix, const float* src_ptr, size_t src_pitch, float* dst_ptr, size_t dst_pitch, size_t height, size_t width) {

    // Computing target positions
    unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int z = threadIdx.z;

    // Out of bounds guard
    if ((x >= width) || (y >= height)) {
        return;
    }

    unsigned int src_pos = y * src_pitch / sizeof(float) + x * 3;
    unsigned int dst_pos = y * dst_pitch  / sizeof(float) + x * 3 + z;

    // Indexing to the z'th row of the transformation matrix M for simplicity
    const float* M_row = m_matrix + z * 3;

    // X = R*M_i,1 + R*M_i,2 + G*M_i,3
    dst_ptr[dst_pos] = src_ptr[src_pos] * M_row[0] + src_ptr[src_pos + 1] * M_row[1]+ src_ptr[src_pos + 2] * M_row[2];
}

/**
 * This function applies the give transformation matrix to perform a colour space transformation.
 * @param m_matrix Pointer to the transformation matrix in device memory
 * @param src_ptr Pointer to the src image in device memory
 * @param src_pitch Pitch of the src image in device memory
 * @param dst_ptr Pointer to the dst image in device memory
 * @param dst_pitch Pitch of the dst image in device memory
 * @param height Height of the image
 * @param width Width of the image
 */
void colour_space_conversion(const float* m_matrix, const float* src_ptr, size_t src_pitch, float* dst_ptr, size_t dst_pitch, size_t height, size_t width) {
    // Creating threads
    dim3 blockDims(16, 16, 3);
    dim3 gridDims((width + blockDims.x - 1) / blockDims.x, (height + blockDims.y - 1) / blockDims.y, 1);

    // Executing kernal
    colour_space_conversion_kernel<<<gridDims, blockDims>>>(m_matrix, src_ptr, src_pitch, dst_ptr, dst_pitch, height, width);

    // Forcing completion
    cudaDeviceSynchronize(); // Synchronize to catch kernel launch errors
    check_cuda_error(cudaGetLastError(), "Colour space conversion kernel launch failed");

}

/**
 * Kernel of the channel flipping transformation
 * @param src_ptr Pointer to src image in device memory
 * @param src_pitch Pitch of the src image in device memory
 * @param dst_ptr Pointer to the allocated dst memory
 * @param dst_pitch Pitch of the allocated dst memory
 * @param height Height of the image
 * @param width Width of the image
 */
__global__ void flip_channel_ordering_kernel(const uint8_t* src_ptr, size_t src_pitch, uint8_t* dst_ptr, size_t dst_pitch, size_t height, size_t width) {

    // Computing target positions
    unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int z = threadIdx.z;

    // Out of bounds guard
    if ((x >= width) || (y >= height)) {
        return;
    }

    // Flipping the channel so that R->B, B->B and B->R (or vice versa depending on the input ordering)
    unsigned int src_pos;
    if (z == 0) {
        src_pos = y * src_pitch / sizeof(uint8_t) + x * 3 + 2;
    } else if (z == 1) {
        src_pos = y * src_pitch / sizeof(uint8_t) + x * 3 + 1;
    } else {
        src_pos = y * src_pitch / sizeof(uint8_t) + x * 3;
    }

    unsigned int dst_pos = y * dst_pitch / sizeof(uint8_t) + x * 3 + z;

    dst_ptr[dst_pos] = src_ptr[src_pos];
}

/**
 * Function that flips channel order from RGB->BGR OR BGR->RGB
 * @param src_ptr Pointer to src image in device memory.  Image is expected to have values in [0,255]
 * @param src_pitch Pitch of the src image in device memory
 * @param dst_ptr Pointer to the allocated dst memory
 * @param dst_pitch Pitch of the allocated dst memory
 * @param height Height of the image
 * @param width Width of the image
 */
void flip_channel_ordering(const uint8_t* src_ptr, size_t src_pitch, uint8_t* dst_ptr, size_t dst_pitch, size_t height, size_t width) {
    // Creating threads
    dim3 blockDims(16, 16, 3);
    dim3 gridDims((width + blockDims.x - 1) / blockDims.x, (height + blockDims.y - 1) / blockDims.y, 1);

    // Executing kernel
    flip_channel_ordering_kernel<<<gridDims, blockDims>>>(src_ptr, src_pitch, dst_ptr, dst_pitch, height, width);

    // Forcing completion
    cudaDeviceSynchronize(); // Synchronize to catch kernel launch errors
    check_cuda_error(cudaGetLastError(), "Flip channel ordering kernel launch failed");

}
