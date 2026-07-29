#include <stdio.h>

#include "tensor.h"

int test_scalar_plus_tensor(void) {
    int shapeA[1] = {3};
    int shapeB[1] = {1};

    tensor_t* A = tensor_create(DTYPE_FLOAT32, 1, shapeA);
    tensor_t* B = tensor_create(DTYPE_FLOAT32, 1, shapeB);

    float* a = (float*)A->data;
    float* b = (float*)B->data;

    a[0] = 1;
    a[1] = 2;
    a[2] = 3;
    b[0] = 10;

    tensor_t* C = tensor_add(A, B);
    if (!C) return 1;

    float* c = (float*)C->data;

    if (c[0] != 11 || c[1] != 12 || c[2] != 13) return 1;

    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    return 0;
}

int test_vector_plus_matrix(void) {
    int shapeA[2] = {2, 3};
    int shapeB[1] = {3};

    tensor_t* A = tensor_create(DTYPE_FLOAT32, 2, shapeA);
    tensor_t* B = tensor_create(DTYPE_FLOAT32, 1, shapeB);

    float* a = (float*)A->data;
    float* b = (float*)B->data;

    // A = [[1,2,3],[4,5,6]]
    for (int i = 0; i < 6; ++i) a[i] = i + 1;

    // B = [10,20,30]
    b[0] = 10;
    b[1] = 20;
    b[2] = 30;

    tensor_t* C = tensor_add(A, B);
    if (!C) return 1;

    float* c = (float*)C->data;

    // Expected:
    // [11,22,33]
    // [14,25,36]
    if (c[0] != 11 || c[1] != 22 || c[2] != 33) return 1;
    if (c[3] != 14 || c[4] != 25 || c[5] != 36) return 1;

    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    return 0;
}

int test_row_vector_plus_matrix(void) {
    int shapeA[2] = {2, 3};
    int shapeB[2] = {1, 3};

    tensor_t* A = tensor_create(DTYPE_FLOAT32, 2, shapeA);
    tensor_t* B = tensor_create(DTYPE_FLOAT32, 2, shapeB);

    float* a = (float*)A->data;
    float* b = (float*)B->data;

    for (int i = 0; i < 6; ++i) a[i] = i + 1;

    b[0] = 100;
    b[1] = 200;
    b[2] = 300;

    tensor_t* C = tensor_add(A, B);
    if (!C) return 1;

    float* c = (float*)C->data;

    if (c[0] != 101 || c[1] != 202 || c[2] != 303) return 1;
    if (c[3] != 104 || c[4] != 205 || c[5] != 306) return 1;

    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    return 0;
}

int test_incompatible_shapes(void) {
    int shapeA[1] = {3};
    int shapeB[1] = {4};

    tensor_t* A = tensor_create(DTYPE_FLOAT32, 1, shapeA);
    tensor_t* B = tensor_create(DTYPE_FLOAT32, 1, shapeB);

    tensor_t* C = tensor_add(A, B);

    tensor_free(A);
    tensor_free(B);

    return (C != NULL);  // debe ser NULL → return 0
}

int main(void) {
    if (test_scalar_plus_tensor()) {
        printf("FAIL: scalar + tensor\n");
        return 1;
    }

    if (test_vector_plus_matrix()) {
        printf("FAIL: vector + matrix\n");
        return 1;
    }

    if (test_row_vector_plus_matrix()) {
        printf("FAIL: row vector + matrix\n");
        return 1;
    }

    if (test_incompatible_shapes()) {
        printf("FAIL: incompatible shapes\n");
        return 1;
    }

    printf("PASS: broadcasting\n");
    return 0;
}
