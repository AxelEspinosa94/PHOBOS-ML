#include <stdio.h>

#include "tensor.h"

int main(void) {
    int shape[1] = {3};
    tensor_t* A = tensor_create(DTYPE_FLOAT32, 1, shape);
    tensor_t* B = tensor_create(DTYPE_FLOAT32, 1, shape);

    float* ad = (float*)A->data;
    float* bd = (float*)B->data;

    ad[0] = 1.0f;
    ad[1] = 2.0f;
    ad[2] = 3.0f;
    bd[0] = 4.0f;
    bd[1] = 5.0f;
    bd[2] = 6.0f;

    tensor_t* C_add = tensor_add(A, B);
    tensor_t* C_sub = tensor_sub(A, B);
    tensor_t* C_mul = tensor_mul(A, B);

    if (!C_add || !C_sub || !C_mul) {
        printf("FAIL: elemwise ops returned NULL\n");
        return 1;
    }

    float* add = (float*)C_add->data;
    float* sub = (float*)C_sub->data;
    float* mul = (float*)C_mul->data;

    if (add[0] != 5.0f || add[1] != 7.0f || add[2] != 9.0f) {
        printf("FAIL: tensor_add incorrect\n");
        return 1;
    }

    if (sub[0] != -3.0f || sub[1] != -3.0f || sub[2] != -3.0f) {
        printf("FAIL: tensor_sub incorrect\n");
        return 1;
    }

    if (mul[0] != 4.0f || mul[1] != 10.0f || mul[2] != 18.0f) {
        printf("FAIL: tensor_mul incorrect\n");
        return 1;
    }

    printf("PASS: elemwise ops\n");

    tensor_free(A);
    tensor_free(B);
    tensor_free(C_add);
    tensor_free(C_sub);
    tensor_free(C_mul);
    return 0;
}
