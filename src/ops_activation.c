/* ops_activation.c — PHOBOS-ML activation ops */
#include <math.h>    // expf
#include <stddef.h>  // size_t

#include "tensor.h"

/* =========================
   tensor_copy
   ========================= */

tensor_t* tensor_copy(const tensor_t* src) {
    if (!src) return NULL;

    // Allocate new tensor with same shape
    tensor_t* t = tensor_create(src->dtype, src->ndim, src->shape);
    if (!t) return NULL;

    // Copy raw data (float32 only for now)
    size_t bytes = src->size * sizeof(float);
    memcpy(t->data, src->data, bytes);

    return t;
}

/* =========================
   ReLU: max(x, 0)
   ========================= */

tensor_t* tensor_relu(const tensor_t* A) {
    if (!A) return NULL;
    if (A->dtype != DTYPE_FLOAT32) return NULL;

    tensor_t* C = tensor_copy(A);
    if (!C) return NULL;

    float* cd = (float*)C->data;

    for (size_t i = 0; i < C->size; ++i) {
        float x = cd[i];
        cd[i] = (x > 0.0f) ? x : 0.0f;
    }

    return C;
}

/* =========================
   Sigmoid: numerically stable
   ========================= */

static float sigmoid_stable(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    } else {
        float e = expf(x);
        return e / (1.0f + e);
    }
}

tensor_t* tensor_sigmoid(const tensor_t* A) {
    if (!A) return NULL;
    if (A->dtype != DTYPE_FLOAT32) return NULL;

    tensor_t* C = tensor_copy(A);
    if (!C) return NULL;

    float* cd = (float*)C->data;

    for (size_t i = 0; i < C->size; ++i) {
        cd[i] = sigmoid_stable(cd[i]);
    }

    return C;
}

/* =========================
   Softmax: last dimension, overflow-safe
   ========================= */

tensor_t* tensor_softmax(const tensor_t* A) {
    if (!A) return NULL;
    if (A->dtype != DTYPE_FLOAT32) return NULL;
    if (A->ndim < 1) return NULL;

    int last_dim = A->shape[A->ndim - 1];
    if (last_dim <= 0) return NULL;

    tensor_t* C = tensor_copy(A);
    if (!C) return NULL;

    float* cd = (float*)C->data;

    size_t total = C->size;
    size_t outer = total / (size_t)last_dim;

    for (size_t o = 0; o < outer; ++o) {
        float* row = cd + o * (size_t)last_dim;

        /* 1. max(x) for stability */
        float maxv = row[0];
        for (int i = 1; i < last_dim; ++i) {
            if (row[i] > maxv) maxv = row[i];
        }

        /* 2. exp(x - max) and sum */
        float sum = 0.0f;
        for (int i = 0; i < last_dim; ++i) {
            row[i] = expf(row[i] - maxv);
            sum += row[i];
        }

        /* 3. normalize */
        if (sum > 0.0f) {
            for (int i = 0; i < last_dim; ++i) {
                row[i] /= sum;
            }
        }
    }

    return C;
}
