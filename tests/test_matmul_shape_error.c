#include <stdio.h>

#include "tensor.h"

int main(void) {
    int A_shape[2] = {2, 3};
    int B_shape[2] = {4, 2};

    tensor_t* A = tensor_create(DTYPE_FLOAT32, 2, A_shape);
    tensor_t* B = tensor_create(DTYPE_FLOAT32, 2, B_shape);

    tensor_t* C = tensor_matmul(A, B);

    if (C != NULL) {
        printf("FAIL: matmul should return NULL on shape mismatch\n");
        return 1;
    }

    printf("PASS: matmul shape mismatch\n");

    tensor_free(A);
    tensor_free(B);
    return 0;
}
