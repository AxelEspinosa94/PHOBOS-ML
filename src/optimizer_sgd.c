#include <stdlib.h>

#include "optimizer.h"

int optimizer_sgd_step(
    tensor_t* W,
    float* b,
    const tensor_t* dW,
    float db,
    float lr) {
    if (!W || !b || !dW)
        return -1;

    if (W->size != dW->size)
        return -2;

    float* Wd = (float*)W->data;
    float* dWd = (float*)dW->data;

    for (size_t i = 0; i < W->size; ++i)
        Wd[i] -= lr * dWd[i];

    *b -= lr * db;

    return 0;
}
