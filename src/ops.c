/* ops.c — PHOBOS-ML operations module */
#include <stdio.h>   // printf
#include <stdlib.h>  // malloc, free
#include <string.h>  // memcpy

#include "tensor.h"

/* =========================
   MatMul (no broadcasting)
   ========================= */

tensor_t* tensor_matmul(const tensor_t* A, const tensor_t* B) {
    if (!A || !B) return NULL;

    if (A->ndim != 2 || B->ndim != 2) {
        return NULL;
    }

    if (A->dtype != DTYPE_FLOAT32 || B->dtype != DTYPE_FLOAT32) {
        return NULL;  // por ahora solo soportamos float32
    }

    int m = A->shape[0];
    int kA = A->shape[1];
    int kB = B->shape[0];
    int n = B->shape[1];

    if (kA != kB) {
        return NULL;
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

/* =========================
   Elemwise (same-shape helper)
   ========================= */

static int tensor_same_shape(const tensor_t* A, const tensor_t* B) {
    if (!A || !B) return 0;
    if (A->ndim != B->ndim) return 0;
    for (int i = 0; i < A->ndim; ++i) {
        if (A->shape[i] != B->shape[i]) return 0;
    }
    return 1;
}

static tensor_t* tensor_elemwise_alloc_like(const tensor_t* A) {
    int* shape = A->shape;
    tensor_t* C = tensor_create(A->dtype, A->ndim, shape);
    return C;
}

/* =========================
   Broadcasting helpers
   ========================= */

static int broadcast_shapes(const tensor_t* A, const tensor_t* B,
                            int out_shape[], int* out_ndim) {
    int ndimA = A->ndim;
    int ndimB = B->ndim;
    int ndim = (ndimA > ndimB) ? ndimA : ndimB;

    for (int i = 0; i < ndim; ++i) {
        int dimA = (i < ndimA) ? A->shape[ndimA - 1 - i] : 1;
        int dimB = (i < ndimB) ? B->shape[ndimB - 1 - i] : 1;

        if (dimA != dimB && dimA != 1 && dimB != 1) {
            return 0;  // incompatible
        }

        out_shape[ndim - 1 - i] = (dimA > dimB) ? dimA : dimB;
    }

    *out_ndim = ndim;
    return 1;
}

static void compute_broadcast_strides(const tensor_t* T,
                                      const int out_shape[],
                                      int out_ndim,
                                      size_t out_strides[]) {
    int offset = out_ndim - T->ndim;
    size_t elem_size = tensor_dtype_size(T->dtype);

    for (int i = 0; i < out_ndim; ++i) {
        int Ti = i - offset;

        if (Ti < 0) {
            out_strides[i] = 0;
        } else if (T->shape[Ti] == 1) {
            out_strides[i] = 0;
        } else {
            out_strides[i] = T->strides[Ti] / elem_size;  // <-- bytes -> elementos
        }
    }
}

static tensor_t* tensor_elemwise_broadcast(
    const tensor_t* A, const tensor_t* B,
    float (*op)(float, float)) {
    if (!A || !B) return NULL;

    if (A->dtype != DTYPE_FLOAT32 || B->dtype != DTYPE_FLOAT32) {
        return NULL;  // por ahora solo soportamos float32
    }

    int out_shape[8];
    int out_ndim;

    if (!broadcast_shapes(A, B, out_shape, &out_ndim)) {
        return NULL;
    }

    tensor_t* C = tensor_create(DTYPE_FLOAT32, out_ndim, out_shape);
    if (!C) return NULL;

    size_t stridesA[8], stridesB[8];
    compute_broadcast_strides(A, out_shape, out_ndim, stridesA);
    compute_broadcast_strides(B, out_shape, out_ndim, stridesB);

    float* ad = (float*)A->data;
    float* bd = (float*)B->data;
    float* cd = (float*)C->data;

    size_t total = C->size;

    for (size_t idx = 0; idx < total; ++idx) {
        size_t offsetA = 0, offsetB = 0;

        size_t tmp = idx;
        for (int d = out_ndim - 1; d >= 0; --d) {
            int coord = tmp % out_shape[d];
            tmp /= out_shape[d];

            offsetA += coord * stridesA[d];
            offsetB += coord * stridesB[d];
        }

        cd[idx] = op(ad[offsetA], bd[offsetB]);
    }

    return C;
}

/* =========================
   Public elemwise ops (with broadcasting)
   ========================= */

static float op_add(float a, float b) { return a + b; }
static float op_sub(float a, float b) { return a - b; }
static float op_mul(float a, float b) { return a * b; }

tensor_t* tensor_add(const tensor_t* A, const tensor_t* B) {
    return tensor_elemwise_broadcast(A, B, op_add);
}

tensor_t* tensor_sub(const tensor_t* A, const tensor_t* B) {
    return tensor_elemwise_broadcast(A, B, op_sub);
}

tensor_t* tensor_mul(const tensor_t* A, const tensor_t* B) {
    return tensor_elemwise_broadcast(A, B, op_mul);
}

tensor_t* tensor_logreg_forward(const tensor_t* X,
                                const tensor_t* W,
                                const tensor_t* b) {
    if (!X || !W) return NULL;

    // X: (m × n)
    // W: (n × 1)
    if (X->ndim != 2 || W->ndim != 2) return NULL;
    if (X->shape[1] != W->shape[0]) return NULL;
    if (W->shape[1] != 1) return NULL;  // W must be column vector

    // 1. Z = XW
    tensor_t* Z = tensor_matmul(X, W);
    if (!Z) return NULL;

    tensor_t* Zb = NULL;

    // 2. Add bias if provided
    if (b) {
        Zb = tensor_add(Z, b);  // broadcasting handles (1×1) or (m×1)
        tensor_free(Z);
        if (!Zb) return NULL;
    } else {
        Zb = Z;  // no bias
    }

    // 3. Apply sigmoid
    tensor_t* Y = tensor_sigmoid(Zb);
    tensor_free(Zb);

    return Y;
}

float tensor_sum(const tensor_t* A) {
    if (!A || A->dtype != DTYPE_FLOAT32) return 0.0f;

    float* ad = (float*)A->data;
    float s = 0.0f;

    for (size_t i = 0; i < A->size; ++i)
        s += ad[i];

    return s;
}

float tensor_mean(const tensor_t* A) {
    if (!A || A->dtype != DTYPE_FLOAT32) return 0.0f;
    if (A->size == 0) return 0.0f;

    return tensor_sum(A) / (float)A->size;
}

float tensor_max(const tensor_t* A) {
    if (!A || A->dtype != DTYPE_FLOAT32) return 0.0f;
    if (A->size == 0) return 0.0f;

    float* ad = (float*)A->data;
    float m = ad[0];

    for (size_t i = 1; i < A->size; ++i)
        if (ad[i] > m) m = ad[i];

    return m;
}
