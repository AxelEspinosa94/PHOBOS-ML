#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"

/**
 * @brief Basic SGD update rule.
 *
 * Updates:
 *   W := W - lr * dW
 *   b := b - lr * db
 *
 * @param W     Weights (n × 1), updated in-place
 * @param b     Bias (scalar), updated in-place
 * @param dW    Gradient of weights (n × 1)
 * @param db    Gradient of bias (scalar)
 * @param lr    Learning rate
 *
 * @return 0 on success, non-zero on error.
 */
int optimizer_sgd_step(
    tensor_t* W,
    float* b,
    const tensor_t* dW,
    float db,
    float lr);

#endif
