#include "tensor.h"
#include <stdio.h>

int main(void) {
    int shape[2] = {2, 3};

    tensor_t* a = tensor_create(DTYPE_FLOAT32, 2, shape);
    tensor_t* b = tensor_create(DTYPE_FLOAT32, 2, shape);

    float* ad = (float*)a->data;
    float* bd = (float*)b->data;

    for (size_t i = 0; i < a->size; ++i) {
        ad[i] = (float)i;
        bd[i] = (float)i;
    }

    if (!tensor_compare(a, b)) {
        printf("FAIL: tensor_compare should be equal\n");
        return 1;
    }

    bd[0] = 999.0f;

    if (tensor_compare(a, b)) {
        printf("FAIL: tensor_compare should detect difference\n");
        return 1;
    }

    printf("PASS: tensor_compare\n");

    tensor_free(a);
    tensor_free(b);
    return 0;
}
