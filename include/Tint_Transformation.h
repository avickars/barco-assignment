/*!
@file Tint_Transformation.h

@brief Header file containing declarations of the Tint_Transformation class

### Author(s)
- Created by Aidan Vickars on Dec 13, 2023

*/


#ifndef BARCO_TINT_TRANSFORMATION_H
#define BARCO_TINT_TRANSFORMATION_H

#include <iostream>

class Tint_Transformation {

private:
    size_t HEIGHT; /// Height of the image
    size_t WIDTH; /// Width of the image

    // Img device variables
    void* device_img_ptr = nullptr; /// Pointer to the image in device memory
    size_t device_img_pitch; /// Pitch of the device_img_ptr memory

    // Img device variables
    void* device_rgb_ptr = nullptr; /// Pointer to the image in RGB format in device memory
    size_t device_rgb_pitch; /// Pitch of the device_rgb_ptr memory

    // Tint Device variable
    void* tint_device_ptr = nullptr; /// Pointer to the tint in device memory

public:
    /**
     * @brief Constructor
     */
    Tint_Transformation(size_t, size_t, const uint8_t*);

    /**
     * @brief Destructor
     */
    ~Tint_Transformation();

    /**
     * Method that applies tint to the image
     */
    void tint_transformation(uint8_t*, size_t, float);
};


#endif //BARCO_TINT_TRANSFORMATION_H
