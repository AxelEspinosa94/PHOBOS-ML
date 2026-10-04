#include <stdio.h>

#include "tensor.h"
#include "train.h"

int main(void) {
    int x_shape[2] = {4, 2};

    tensor_t* X =
        tensor_create(
            DTYPE_FLOAT32,
            2,
            x_shape);

    float* xd = (float*)X->data;

    xd[0] = 0;
    xd[1] = 0;
    xd[2] = 0;
    xd[3] = 1;
    xd[4] = 1;
    xd[5] = 0;
    xd[6] = 1;
    xd[7] = 1;

    int y_shape[2] = {4, 1};

    tensor_t* y =
        tensor_create(
            DTYPE_FLOAT32,
            2,
            y_shape);

    float* yd = (float*)y->data;

    yd[0] = 0;
    yd[1] = 0;
    yd[2] = 1;
    yd[3] = 1;

    int w_shape[2] = {2, 1};

    tensor_t* W =
        tensor_create(
            DTYPE_FLOAT32,
            2,
            w_shape);

    float* wd = (float*)W->data;

    wd[0] = 0.0f;
    wd[1] = 0.0f;

    float b = 0.0f;

    int err =
        train_loop(
            X,
            y,
            W,
            &b,
            0.1f,
            10);

    if (err != 0) {
        printf("FAIL: training loop\n");
        return 1;
    }

    if (wd[0] == 0.0f &&
        wd[1] == 0.0f) {
        printf("FAIL: weights unchanged\n");
        return 1;
    }

    printf(
        "PASS: training loop\n");

    tensor_free(X);
    tensor_free(y);
    tensor_free(W);

    return 0;
}