//
// Created by aidan on 12/12/23.
//

#include "Colour_Space_Transformation.h"
#include "cuda_utils.cuh"
#include "cuda_transformations.cuh"

static const float R709_TO_R2020_TRANSFORMATION_MATRIX[] = {
        0.6274039,  0.32928304, 0.04331307,
        0.06909729, 0.9195404,  0.01136232,
        0.01639144, 0.08801331, 0.89559525
}; /// Transition matrix to transform an r709 RGB pixel into r2020 RGB pixel

/**
 * Constructor that pre-allocates all device memory needed for the transformation
 * @param height Height of the input image
 * @param width Width of the input image
 */
Colour_Space_Transformation::Colour_Space_Transformation(size_t height, size_t width): HEIGHT(height), WIDTH(width) {

    // Allocating device memory to hold the original image
    allocate_2d_device_memory(this->device_img_ptr, this->device_img_pitch, sizeof(uint8_t) * this->WIDTH* 3, this->HEIGHT);

    // Allocating device memory to hold the image in RGB order
    allocate_2d_device_memory(this->device_rgb_ptr, this->device_rgb_pitch, sizeof(uint8_t) * this->WIDTH * 3, this->HEIGHT);

    // Allocating device memory to hold the normalized image
    allocate_2d_device_memory(this->device_img_nml_ptr, this->device_img_nml_pitch, sizeof(float) * this->WIDTH* 3, this->HEIGHT);

    // Allocating device memory to hold the transformed image
    allocate_2d_device_memory(this->device_img_r2020_ptr, this->device_img_r2020_pitch, sizeof(float) * this->WIDTH * 3, this->HEIGHT);

    allocate_device_memory(this->device_transformation_matrix_ptr, sizeof(float) * 9);
    copy((void*)this->device_transformation_matrix_ptr, (void*) R709_TO_R2020_TRANSFORMATION_MATRIX, sizeof(float) * 9, cudaMemcpyHostToDevice);

}

/**
 * Destructor the releases all memory allocated in the constructor
 */
Colour_Space_Transformation::~Colour_Space_Transformation() {

    // Deallocating device memory that holds the original image
    deallocate_device_memory(this->device_img_ptr);

    // Deallocating Allocating device memory to hold the image in RGB order
    deallocate_device_memory(this->device_rgb_ptr);

    // Deallocating device memory that holds the normalized image
    deallocate_device_memory(this->device_img_nml_ptr);

    // Deallocating device memory that holds the transformed image
    deallocate_device_memory(this->device_img_r2020_ptr);

    // Deallocating device memory that hols the transformation matrix
    deallocate_device_memory(this->device_transformation_matrix_ptr);


}

/**
 * Method that transforms an image from r709 colour space to r2020 colour space.  The transformation is done
 * using an in-place operation
 * @param img_ptr Pointer to the raw image in host memory
 * @param img_pitch Pitch of the raw image (in-bytes) img_ptr in host memory
 */
void Colour_Space_Transformation::r709_to_r2020(uint8_t *img_ptr, size_t img_pitch) {

    // Copying image to device memory
    copy_2d(
            this->device_img_ptr,
            this->device_img_pitch,
            img_ptr,
            img_pitch,
            sizeof(uint8_t) * this->WIDTH* 3,
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

    // Normalizing the image to [0,1]
    normalize(
            (uint8_t*) this->device_rgb_ptr,
            this->device_rgb_pitch,
            (float*) this->device_img_nml_ptr,
            this->device_img_nml_pitch,
            this->HEIGHT,
            this->WIDTH
    );

    // Performing the colour space conversion
    colour_space_conversion(
            (float*) this->device_transformation_matrix_ptr,
            (float*) this->device_img_nml_ptr,
            this->device_img_nml_pitch,
            (float*)  this->device_img_r2020_ptr,
            this->device_img_r2020_pitch,
            this->HEIGHT,
            this->WIDTH
    );

    // We re-use the memory allocated to device_rgb_ptr as it is no longer needed
    // and denormalize the image back to [0,255]
    denormalize(
            (float*)  this->device_img_r2020_ptr,
            this->device_img_r2020_pitch,
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

    // Copying the transformed image back to its original location
    copy_2d(
            img_ptr,
            img_pitch,
            this->device_img_ptr,
            this->device_img_pitch,
            sizeof(uint8_t)*this->WIDTH*3,
            this->HEIGHT,
            cudaMemcpyDeviceToHost
    );
}

