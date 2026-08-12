#include <math.h>
#include <stdio.h>

#include "tensor.h"

int main(void) {
    // X: 2×2
    int shapeX[2] = {2, 2};
    tensor_t* X = tensor_create(DTYPE_FLOAT32, 2, shapeX);
    float* xd = (float*)X->data;
    xd[0] = 1;
    xd[1] = 2;
    xd[2] = 3;
    xd[3] = 4;

    // y: 2×1
    int shapeY[2] = {2, 1};
    tensor_t* y = tensor_create(DTYPE_FLOAT32, 2, shapeY);
    float* yd = (float*)y->data;
    yd[0] = 1.0f;
    yd[1] = 0.0f;

    // W: 2×1
    int shapeW[2] = {2, 1};
    tensor_t* W = tensor_create(DTYPE_FLOAT32, 2, shapeW);
    float* Wd = (float*)W->data;
    Wd[0] = 0.1f;
    Wd[1] = -0.2f;

    float b = 0.05f;
    float lr = 0.1f;
    float loss = 0.0f;

    int err = tensor_logreg_train_step(X, y, W, &b, lr, &loss);
    if (err != 0) {
        printf("FAIL: train step error\n");
        return 1;
    }

    // Expected behavior:
    // Loss decreases and weights move in correct direction.
    // We don't check exact values, only that they changed.

    if (fabsf(Wd[0] - 0.1f) < 1e-6f) return 1;
    if (fabsf(Wd[1] + 0.2f) < 1e-6f) return 1;
    if (fabsf(b - 0.05f) < 1e-6f) return 1;

    printf("PASS: training step\n");
    return 0;
}
