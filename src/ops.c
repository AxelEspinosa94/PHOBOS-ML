#include "tensor.h"

#include <stdlib.h>
#include <string.h>

tensor_t* tensor_matmul(const tensor_t* A, const tensor_t* B) {
    if (!A || !B) return NULL;

    if (A->ndim != 2 || B->ndim != 2) {
        return NULL;// Only 2D matmul supported for now
    }

    int m = A->shape[0];
    int kA = A->shape[1];
    int kB = B->shape[0];
    int n = B->shape[1];

    if (kA != kB) {
        return NULL;// Shape mismatch
    }

    int out_shape[2] = {m, n};
    tensor_t* C = tensor_create(A->dtype, 2, out_shape);
    if (!C) return NULL;

    float* a = (float*)A->data;
    float* b = (float*)B->data;
    float* c = (float*)C->data;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < kA; ++k) {
                sum += a[i * kA + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }

    return C;
}
