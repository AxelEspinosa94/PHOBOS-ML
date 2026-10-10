#ifndef MODEL_IO_H
#define MODEL_IO_H

#include "tensor.h"

/**
 * @brief Save a logistic regression model to disk.
 *
 * Stores:
 *   - magic number
 *   - tensor dimensions
 *   - tensor weights
 *   - bias
 *
 * @param path Output file path.
 * @param W    Weight tensor.
 * @param b    Bias scalar.
 *
 * @return 0 on success, non-zero on failure.
 */
int model_save(
    const char* path,
    const tensor_t* W,
    float b);

/**
 * @brief Load a logistic regression model from disk.
 *
 * @param path Input file path.
 * @param W    Existing weight tensor to populate.
 * @param b    Output bias.
 *
 * @return 0 on success, non-zero on failure.
 */
int model_load(
    const char* path,
    tensor_t* W,
    float* b);

#endif