#include <stdlib.h>

#include "tensor.h"

int tensor_logreg_train_step(
    const tensor_t* X,
    const tensor_t* y,
    tensor_t* W,
    float* b,
    float lr,
    float* loss_out) {
    if (!X || !y || !W || !b || !loss_out)
        return -1;

    // 1. Forward pass: y_hat = sigmoid(XW + b)
    tensor_t* y_hat = tensor_logreg_forward(X, W, NULL);
    if (!y_hat) return -2;

    // Add bias manually (broadcasting)
    float* yhd = (float*)y_hat->data;
    for (size_t i = 0; i < y_hat->size; ++i)
        yhd[i] += *b;

    // Apply sigmoid again after adding bias
    tensor_t* y_hat_final = tensor_sigmoid(y_hat);
    tensor_free(y_hat);
    if (!y_hat_final) return -3;

    // 2. Compute loss
    float loss = tensor_bce_loss(y, y_hat_final);
    *loss_out = loss;

    // 3. Compute gradients
    tensor_t* dW = NULL;
    float db = 0.0f;

    int err = tensor_logreg_gradients(X, y, y_hat_final, &dW, &db);
    tensor_free(y_hat_final);

    if (err != 0) return -4;

    // 4. SGD update
    float* Wd = (float*)W->data;
    float* dWd = (float*)dW->data;

    for (size_t j = 0; j < W->size; ++j)
        Wd[j] -= lr * dWd[j];

    *b -= lr * db;

    tensor_free(dW);

    return 0;
}
