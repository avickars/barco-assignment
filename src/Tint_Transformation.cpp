/*!
@file Tint_Transformation.cpp

@brief Source file containing method definitions of the Tint_Transformation class

### Author(s)
- Created by Aidan Vickars on Dec 13, 2023

*/

#include "Tint_Transformation.h"
#include "cuda_utils.cuh"
#include "cuda_transformations.cuh"


/**
 * Constructor
 * @param height Height of the input image
 * @param width Width of the input image
 * @param tint Tine to apply to the input image.  Expects a 1*3 array representing the tint to apply to each channel
 */
Tint_Transformation::Tint_Transformation(size_t height, size_t width, const uint8_t* tint): HEIGHT(height), WIDTH(width) {

    // Allocating device memory to hold the image
    allocate_2d_device_memory(this->device_img_ptr, this->device_img_pitch, sizeof(uint8_t) * this->WIDTH * 3, this->HEIGHT);

    // Allocating device memory to hold the image in RGB order
    allocate_2d_device_memory(this->device_rgb_ptr, this->device_rgb_pitch, sizeof(uint8_t) * this->WIDTH * 3, this->HEIGHT);

    // Allocating device memory to hold the desired tint
    allocate_device_memory(this->tint_device_ptr, sizeof(uint8_t) * 3);

    // Copying tint to device
    copy(tint_device_ptr, (void*) tint, sizeof(uint8_t) * 3, cudaMemcpyHostToDevice);
}

/**
 * Destructor that releases all memory allocated for the transformation
 */
Tint_Transformation::~Tint_Transformation(){

    // Deallocating device memory to hold the image
    deallocate_device_memory(this->device_img_ptr);

    // Deallocating device memory to hold the image in RGB order
    deallocate_device_memory(this->device_rgb_ptr);

    // Deallocating device memory to hold the desired tint
    deallocate_device_memory(this->tint_device_ptr);

}

/**
 * Method that applies a tint transformation
 * @param src_img_ptr Pointer to the image in host memory
 * @param src_img_pitch Pitch of the image in host memory
 * @param weight Weight to use when applying the tint
 */
void Tint_Transformation::tint_transformation(uint8_t* src_img_ptr, size_t src_img_pitch, float weight) {

    // Copying image to device memory
    copy_2d(
            this->device_img_ptr,
            this->device_img_pitch,
            src_img_ptr,
            src_img_pitch,
            sizeof(uint8_t) * this->WIDTH * 3,
            this->HEIGHT,
            cudaMemcpyHostToDevice
    );

    // Converting from BGR to RGB
    flip_channel_ordering(
            (uint8_t*) this->device_img_ptr,
            this->device_img_pitch,
            (uint8_t*) this->device_rgb_ptr,
            this->device_rgb_pitch,
            this->HEIGHT,
            this->WIDTH
    );


    // Applying the tint
    tint(
            (uint8_t*) this->tint_device_ptr,
            weight,
            (uint8_t*) this->device_rgb_ptr,
            this->device_rgb_pitch,
            this->HEIGHT,
            this->WIDTH
    );

    // Converting from RGB to BGR
    flip_channel_ordering(
            (uint8_t*) this->device_rgb_ptr,
            this->device_rgb_pitch,
            (uint8_t*) this->device_img_ptr,
            this->device_img_pitch,
            this->HEIGHT,
            this->WIDTH
    );

    // Copying the tinted image back to the host
    copy_2d(
            src_img_ptr,
            src_img_pitch,
            this->device_img_ptr,
            this->device_img_pitch,
            sizeof(uint8_t) * this->WIDTH * 3,
            this->HEIGHT,
            cudaMemcpyDeviceToHost
    );
}