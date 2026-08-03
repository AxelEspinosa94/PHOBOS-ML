#include <math.h>
#include <stdio.h>

#include "tensor.h"

int main(void) {
    int shape[1] = {5};
    tensor_t* A = tensor_create(DTYPE_FLOAT32, 1, shape);

    float* ad = (float*)A->data;
    ad[0] = 1.0f;
    ad[1] = 2.0f;
    ad[2] = 3.0f;
    ad[3] = 4.0f;
    ad[4] = 5.0f;

    if (tensor_sum(A) != 15.0f) {
        printf("FAIL: sum\n");
        return 1;
    }

    if (fabsf(tensor_mean(A) - 3.0f) > 1e-6f) {
        printf("FAIL: mean\n");
        return 1;
    }

    if (tensor_max(A) != 5.0f) {
        printf("FAIL: max\n");
        return 1;
    }

    printf("PASS: reductions\n");
    return 0;
}
