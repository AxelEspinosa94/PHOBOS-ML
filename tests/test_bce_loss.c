#include <stdio.h>
#include <math.h>
#include "tensor.h"

int main(void) {
    // y: ground truth
    int shape[2] = {4, 1};
    tensor_t* y = tensor_create(DTYPE_FLOAT32, 2, shape);
    float* yd = (float*)y->data;
    yd[0] = 1.0f;
    yd[1] = 0.0f;
    yd[2] = 1.0f;
    yd[3] = 0.0f;

    // p: predictions (sigmoid outputs)
    tensor_t* p = tensor_create(DTYPE_FLOAT32, 2, shape);
    float* pd = (float*)p->data;
    pd[0] = 0.9f;
    pd[1] = 0.1f;
    pd[2] = 0.8f;
    pd[3] = 0.2f;

    float loss = tensor_bce_loss(y, p);

    // Expected:
    // BCE = -(1/4) * [ log(0.9) + log(0.9) + log(0.8) + log(0.8) ]
    // ≈ 0.16425
    if (fabsf(loss - 0.16425f) > 1e-4f) {
        printf("FAIL: BCE loss\n");
        return 1;
    }

    printf("PASS: BCE loss\n");
    return 0;
}
