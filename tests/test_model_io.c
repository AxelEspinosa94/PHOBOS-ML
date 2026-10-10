#include <math.h>
#include <stdio.h>

#include "model_io.h"
#include "tensor.h"

int main(void) {
    int shape[2] = {2, 1};

    tensor_t* W =
        tensor_create(
            DTYPE_FLOAT32,
            2,
            shape);

    float* wd =
        (float*)W->data;

    wd[0] = 1.5f;
    wd[1] = -2.5f;

    float bias = 0.75f;

    int err =
        model_save(
            "model.bin",
            W,
            bias);

    if (err != 0) {
        printf("FAIL: save\n");
        return 1;
    }

    wd[0] = 0.0f;
    wd[1] = 0.0f;

    bias = 0.0f;

    err =
        model_load(
            "model.bin",
            W,
            &bias);

    if (err != 0) {
        printf("FAIL: load\n");
        return 1;
    }

    if (fabsf(wd[0] - 1.5f) > 1e-6f)
        return 1;

    if (fabsf(wd[1] + 2.5f) > 1e-6f)
        return 1;

    if (fabsf(bias - 0.75f) > 1e-6f)
        return 1;

    printf("PASS: model save/load\n");

    tensor_free(W);

    return 0;
}