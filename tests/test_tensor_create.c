#include <stdio.h>

#include "tensor.h"

int main(void) {
    int shape[2] = {2, 3};
    tensor_t* t = tensor_create(DTYPE_FLOAT32, 2, shape);

    if (!t) {
        printf("FAIL: tensor_create returned NULL\n");
        return 1;
    }

    if (t->ndim != 2 || t->shape[0] != 2 || t->shape[1] != 3) {
        printf("FAIL: tensor_create shape incorrect\n");
        return 1;
    }

    if (t->size != 6) {
        printf("FAIL: tensor_create size incorrect\n");
        return 1;
    }

    printf("PASS: tensor_create\n");
    tensor_free(t);
    return 0;
}
