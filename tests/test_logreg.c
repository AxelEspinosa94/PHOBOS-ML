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

    // W: 2×1
    int shapeW[2] = {2, 1};
    tensor_t* W = tensor_create(DTYPE_FLOAT32, 2, shapeW);
    float* wd = (float*)W->data;
    wd[0] = 0.5f;
    wd[1] = -0.25f;

    // b: 1×1
    int shapeB[2] = {1, 1};
    tensor_t* B = tensor_create(DTYPE_FLOAT32, 2, shapeB);
    float* bd = (float*)B->data;
    bd[0] = 0.1f;

    tensor_t* Y = tensor_logreg_forward(X, W, B);
    if (!Y) {
        printf("FAIL: logistic regression forward\n");
        return 1;
    }

    float* yd = (float*)Y->data;

    // Expected:
    // Z = XW + b
    // Row 1: 1*0.5 + 2*(-0.25) + 0.1 = 0.1
    // Row 2: 3*0.5 + 4*(-0.25) + 0.1 = 0.6
    // Sigmoid(0.1) ≈ 0.524979
    // Sigmoid(0.6) ≈ 0.645656

    if (fabsf(yd[0] - 0.524979f) > 1e-5f) return 1;
    if (fabsf(yd[1] - 0.645656f) > 1e-5f) return 1;

    printf("PASS: logistic regression forward\n");
    return 0;
}
