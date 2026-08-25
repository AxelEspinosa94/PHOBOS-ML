#include <math.h>
#include <stdio.h>

#include "optimizer.h"
#include "tensor.h"

int main(void) {
    // W: 2×1
    int shapeW[2] = {2, 1};
    tensor_t* W = tensor_create(DTYPE_FLOAT32, 2, shapeW);
    float* Wd = (float*)W->data;
    Wd[0] = 1.0f;
    Wd[1] = -2.0f;

    // dW: 2×1
    tensor_t* dW = tensor_create(DTYPE_FLOAT32, 2, shapeW);
    float* dWd = (float*)dW->data;
    dWd[0] = 0.5f;
    dWd[1] = -1.0f;

    float b = 0.2f;
    float db = 0.3f;
    float lr = 0.1f;

    int err = optimizer_sgd_step(W, &b, dW, db, lr);
    if (err != 0) {
        printf("FAIL: optimizer returned error\n");
        return 1;
    }

    // Expected:
    // W[0] = 1.0 - 0.1*0.5 = 0.95
    // W[1] = -2.0 - 0.1*(-1.0) = -1.9
    // b    = 0.2 - 0.1*0.3 = 0.17

    if (fabsf(Wd[0] - 0.95f) > 1e-6f) return 1;
    if (fabsf(Wd[1] + 1.9f) > 1e-6f) return 1;
    if (fabsf(b - 0.17f) > 1e-6f) return 1;

    printf("PASS: optimizer SGD step\n");
    tensor_free(W);
    tensor_free(dW);
    return 0;
}
