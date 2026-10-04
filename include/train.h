#ifndef TRAIN_H
#define TRAIN_H

#include "tensor.h"

/**
 * @brief Perform a single SGD training step for logistic regression.
 *
 * Computes:
 *   y_hat = sigmoid(XW + b)
 *   loss  = BCE(y, y_hat)
 *   dW, db = gradients
 *
 * Updates:
 *   W := W - lr * dW
 *   b := b - lr * db
 *
 * @param X   Input matrix (m × n)
 * @param y   Labels (m × 1)
 * @param W   Weights (n × 1) — updated in-place
 * @param b   Bias (scalar) — updated in-place
 * @param lr  Learning rate
 * @param loss_out  Output loss (float)
 *
 * @return 0 on success, non-zero on error.
 */
int tensor_logreg_train_step(
    const tensor_t* X,
    const tensor_t* y,
    tensor_t* W,
    float* b,
    float lr,
    float* loss_out);

/**
 * @brief Run a logistic regression training loop.
 *
 * Performs:
 * - Forward
 * - BCE loss
 * - Gradient computation
 * - SGD update
 *
 * @param X Training samples.
 * @param y Labels.
 * @param W Weights (updated in-place).
 * @param b Bias (updated in-place).
 * @param lr Learning rate.
 * @param epochs Number of epochs.
 *
 * @return 0 on success.
 */
int train_loop(
    const tensor_t* X,
    const tensor_t* y,
    tensor_t* W,
    float* b,
    float lr,
    int epochs);

#endif