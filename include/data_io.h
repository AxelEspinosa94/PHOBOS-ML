#ifndef DATA_IO_H
#define DATA_IO_H

#include "tensor.h"

/**
 * @brief Load a numeric CSV file into a tensor.
 *
 * All values must be numeric.
 *
 * Example:
 *
 * 1,2,3
 * 4,5,6
 *
 * becomes:
 *
 * Tensor(shape=[2,3])
 *
 * @param path CSV file path.
 * @param skip_header Skip first line if non-zero.
 * @return tensor_t* or NULL on error.
 */
tensor_t* tensor_load_csv(
    const char* path,
    int skip_header);

/**
 * @brief Save a tensor as CSV.
 *
 * Tensor must be 2-dimensional.
 *
 * @param path Output CSV file.
 * @param tensor Tensor to write.
 *
 * @return 0 on success.
 */
int tensor_save_csv(
    const char* path,
    const tensor_t* tensor);

#endif