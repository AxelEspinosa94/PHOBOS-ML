#include <stdio.h>

#include "tensor.h"

int test_matmul_dtype_mismatch(void) {
    int A_shape[2] = {2, 3};
    int B_shape[2] = {3, 2};

    tensor_t* A = tensor_create(DTYPE_INT32, 2, A_shape);
    tensor_t* B = tensor_create(DTYPE_FLOAT32, 2, B_shape);

    if (!A || !B) {
        printf("FAIL: dtype mismatch test setup (alloc failed)\n");
        tensor_free(A);
        tensor_free(B);
        return 1;
    }

    tensor_t* C = tensor_matmul(A, B);

    int failed = (C != NULL);  // debe ser NULL

    tensor_free(A);
    tensor_free(B);
    tensor_free(C);  // tensor_free ya maneja NULL, no rompe nada

    return failed;
}

int main(void) {
    int A_shape[2] = {2, 3};
    int B_shape[2] = {3, 2};

    tensor_t* A = tensor_create(DTYPE_FLOAT32, 2, A_shape);
    tensor_t* B = tensor_create(DTYPE_FLOAT32, 2, B_shape);

    float* a = (float*)A->data;
    float* b = (float*)B->data;

    // A = [1 2 3; 4 5 6]
    a[0] = 1;
    a[1] = 2;
    a[2] = 3;
    a[3] = 4;
    a[4] = 5;
    a[5] = 6;

    // B = [7 8; 9 10; 11 12]
    b[0] = 7;
    b[1] = 8;
    b[2] = 9;
    b[3] = 10;
    b[4] = 11;
    b[5] = 12;

    tensor_t* C = tensor_matmul(A, B);

    if (!C) {
        printf("FAIL: matmul returned NULL\n");
        return 1;
    }

    float* c = (float*)C->data;

    // Expected:
    // [58 64]
    // [139 154]

    if (c[0] != 58 || c[1] != 64 || c[2] != 139 || c[3] != 154) {
        printf("FAIL: matmul incorrect result\n");
        return 1;
    }

    printf("PASS: matmul\n");

    tensor_free(A);
    tensor_free(B);
    tensor_free(C);

    if (test_matmul_dtype_mismatch()) {
        printf("FAIL: matmul dtype mismatch (A=INT32) should return NULL\n");
        return 1;
    }

    printf("PASS: matmul dtype mismatch\n");

    return 0;
}