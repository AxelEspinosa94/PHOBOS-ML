#include "tensor.h"
#include <stdio.h>

int main(void) {
    int shapeA[1] = {3};
    int shapeB[1] = {4};

    tensor_t* A = tensor_create(DTYPE_FLOAT32, 1, shapeA);
    tensor_t* B = tensor_create(DTYPE_FLOAT32, 1, shapeB);

    tensor_t* C = tensor_add(A, B);
    if (C != NULL) {
        printf("FAIL: tensor_add should return NULL on shape mismatch\n");
        return 1;
    }

    printf("PASS: elemwise shape mismatch\n");

    tensor_free(A);
    tensor_free(B);
    return 0;
}
