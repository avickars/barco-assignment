/*!
@file cuda_transformations.cuh

@brief Header file containing function declarations of transformations using cuda

### Author(s)
- Created by Aidan Vickars on Dec 13, 2023

*/

#ifndef BARCO_CUDA_TRANSFORMATIONS_H
#define BARCO_CUDA_TRANSFORMATIONS_H

#include <cstdint>

/**
 * @brief Function that applies a tine to an image in device memory
 */
void tint(const uint8_t*, float, uint8_t*, size_t, size_t, size_t);

/**
 * @brief Function that normalizes an image in device memory
 */
void normalize(const uint8_t*, size_t, float*, size_t, size_t, size_t);

/**
 * @brief Function that denormalizes an image in device memory
 */
void denormalize(const float*, size_t, uint8_t*, size_t, size_t, size_t);

/**
 * @brief Function that applies a colour space conversion
 */
void colour_space_conversion(const float*, const float*, size_t, float*, size_t, size_t, size_t);

/**
 * @brief Flips RGB to BGR order OR BGR to RGB
 */
void flip_channel_ordering(const uint8_t*, size_t, uint8_t*, size_t, size_t, size_t);

#endif //BARCO_CUDA_TRANSFORMATIONS_H
