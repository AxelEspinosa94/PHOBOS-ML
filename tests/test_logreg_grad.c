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

    // y_hat: 2×1
    tensor_t* y_hat = tensor_create(DTYPE_FLOAT32, 2, shapeY);
    float* yh = (float*)y_hat->data;
    yh[0] = 0.8f;
    yh[1] = 0.3f;

    tensor_t* dW = NULL;
    float db = 0.0f;

    int err = tensor_logreg_gradients(X, y, y_hat, &dW, &db);
    if (err != 0) {
        printf("FAIL: gradient computation error\n");
        return 1;
    }

    float* dWd = (float*)dW->data;

    // Manual expected:
    // e = [ -0.2, 0.3 ]
    // dW = X^T e
    // dW[0] = 1*(-0.2) + 3*(0.3) = 0.7
    // dW[1] = 2*(-0.2) + 4*(0.3) = 0.8
    // db = -0.2 + 0.3 = 0.1

    if (fabsf(dWd[0] - 0.7f) > 1e-5f) return 1;
    if (fabsf(dWd[1] - 0.8f) > 1e-5f) return 1;
    if (fabsf(db - 0.1f) > 1e-5f) return 1;

    printf("PASS: logistic regression gradients\n");
    tensor_free(dW);
    return 0;
}
