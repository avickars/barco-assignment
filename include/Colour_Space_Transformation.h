/*!
@file Colour_Space_Transformation.h

@brief Heading file containing Colour_Space_Transformation declarations

### Author(s)
- Created by Aidan Vickars on Dec 13, 2023

*/
#ifndef BARCO_COLOUR_SPACE_TRANSFORMATION_H
#define BARCO_COLOUR_SPACE_TRANSFORMATION_H

#include <iostream>

/**
 * @brief Object defining colour space transformations
 */
class Colour_Space_Transformation {

private:
    const size_t HEIGHT; /// Height of the image
    const size_t WIDTH; /// Width of the image

    // Img device variables
    void* device_img_ptr = nullptr; /// Pointer to device allocated memory to hold the image
    size_t device_img_pitch; /// Pitch (in bytes) of the device_img_ptr

    // Img device variables
    void* device_rgb_ptr = nullptr; /// Pointer to the image in RGB format in device memory
    size_t device_rgb_pitch; /// Pitch of the device_rgb_ptr memory

    // Normalized image device variables
    void* device_img_nml_ptr = nullptr;/// Pointer to device allocated memory to the normalize image
    size_t device_img_nml_pitch; /// Pitch (in bytes) of the device_img_nml_ptr

    // Normalized image device variables
    void* device_img_r2020_ptr = nullptr; /// Pointer to device allocated memory to hold the in r2020 colour space
    size_t device_img_r2020_pitch; /// Pitch (in bytes) of the device_img_r2020_ptr

    // Transition matrix variables
    void* device_transformation_matrix_ptr = nullptr;/// Pointer to device allocated memory to hold the transition matix of of the colour space transformation



public:

    /**
     * @brief Constructor
     */
    Colour_Space_Transformation(size_t, size_t);

    /**
     * @brief Destructor
     */
    ~Colour_Space_Transformation();

    /**
     * @brief Transformation from r709 to r2020 colour space
     */
    void r709_to_r2020(uint8_t*, size_t);

};


#endif //BARCO_COLOUR_SPACE_TRANSFORMATION_H
