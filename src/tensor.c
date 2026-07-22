#include "tensor.h"

#include <stdio.h>   // printf
#include <stdlib.h>  // malloc, free
#include <string.h>  // memcpy

/* ============================================================
   Internal helpers
   ============================================================ */

/**
 * @brief Computes total number of elements from shape.
 */
static size_t tensor_compute_size(int ndim, const int* shape) {
    if (ndim <= 0 || shape == NULL) {
        return 0;
    }

    size_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        if (shape[i] <= 0) {
            return 0;
        }
        size *= (size_t)shape[i];
    }
    return size;
}

/**
 * @brief Computes strides in bytes for a contiguous tensor.
 *
 * Layout: row-major (C-style), last dimension contiguous.
 */
static void tensor_compute_strides(int ndim, const int* shape, size_t elem_size, size_t* strides) {
    if (ndim <= 0 || shape == NULL || strides == NULL) {
        return;
    }

    size_t stride = elem_size;
    for (int i = ndim - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= (size_t)shape[i];
    }
}

/* ============================================================
   Public API implementation
   ============================================================ */

size_t tensor_dtype_size(dtype_t dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32:
            return sizeof(float);
        case DTYPE_FLOAT64:
            return sizeof(double);
        case DTYPE_INT32:
            return sizeof(int32_t);
        case DTYPE_INT64:
            return sizeof(int64_t);
        default:
            return 0;
    }
}

tensor_t* tensor_create(dtype_t dtype, int ndim, const int* shape) {
    if (ndim <= 0 || shape == NULL) {
        return NULL;
    }

    size_t elem_size = tensor_dtype_size(dtype);
    if (elem_size == 0) {
        return NULL;
    }

    size_t size = tensor_compute_size(ndim, shape);
    if (size == 0) {
        return NULL;
    }

    tensor_t* t = (tensor_t*)malloc(sizeof(tensor_t));
    if (t == NULL) {
        return NULL;
    }

    t->ndim = ndim;
    t->dtype = dtype;
    t->size = size;
    t->bytes = size * elem_size;
    t->owns_data = true;

    t->shape = (int*)malloc(sizeof(int) * (size_t)ndim);
    t->strides = (size_t*)malloc(sizeof(size_t) * (size_t)ndim);
    t->data = malloc(t->bytes);

    if (t->shape == NULL || t->strides == NULL || t->data == NULL) {
        if (t->shape) free(t->shape);
        if (t->strides) free(t->strides);
        if (t->data) free(t->data);
        free(t);
        return NULL;
    }

    memcpy(t->shape, shape, sizeof(int) * (size_t)ndim);
    tensor_compute_strides(ndim, shape, elem_size, t->strides);

    return t;
}

tensor_t* tensor_from_data(void* data, dtype_t dtype, int ndim, const int* shape) {
    if (data == NULL || ndim <= 0 || shape == NULL) {
        return NULL;
    }

    size_t elem_size = tensor_dtype_size(dtype);
    if (elem_size == 0) {
        return NULL;
    }

    size_t size = tensor_compute_size(ndim, shape);
    if (size == 0) {
        return NULL;
    }

    tensor_t* t = (tensor_t*)malloc(sizeof(tensor_t));
    if (t == NULL) {
        return NULL;
    }

    t->ndim = ndim;
    t->dtype = dtype;
    t->size = size;
    t->bytes = size * elem_size;
    t->owns_data = false;

    t->shape = (int*)malloc(sizeof(int) * (size_t)ndim);
    t->strides = (size_t*)malloc(sizeof(size_t) * (size_t)ndim);
    t->data = data;

    if (t->shape == NULL || t->strides == NULL) {
        if (t->shape) free(t->shape);
        if (t->strides) free(t->strides);
        free(t);
        return NULL;
    }

    memcpy(t->shape, shape, sizeof(int) * (size_t)ndim);
    tensor_compute_strides(ndim, shape, elem_size, t->strides);

    return t;
}

void tensor_free(tensor_t* t) {
    if (t == NULL) {
        return;
    }

    if (t->owns_data && t->data != NULL) {
        free(t->data);
    }

    if (t->shape != NULL) {
        free(t->shape);
    }

    if (t->strides != NULL) {
        free(t->strides);
    }

    free(t);
}

bool tensor_is_contiguous(const tensor_t* t) {
    if (t == NULL || t->shape == NULL || t->strides == NULL) {
        return false;
    }

    size_t elem_size = tensor_dtype_size(t->dtype);
    if (elem_size == 0) {
        return false;
    }

    size_t expected_stride = elem_size;
    for (int i = t->ndim - 1; i >= 0; --i) {
        if (t->strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= (size_t)t->shape[i];
    }

    return true;
}

void tensor_print(const tensor_t* t) {
    if (t == NULL) {
        printf("tensor_t: NULL\n");
        return;
    }

    printf("tensor_t {\n");
    printf("  dtype: %d\n", (int)t->dtype);
    printf("  ndim: %d\n", t->ndim);
    printf("  size: %zu\n", t->size);
    printf("  bytes: %zu\n", t->bytes);
    printf("  owns_data: %s\n", t->owns_data ? "true" : "false");

    printf("  shape: [");
    for (int i = 0; i < t->ndim; ++i) {
        printf("%d", t->shape[i]);
        if (i + 1 < t->ndim) {
            printf(", ");
        }
    }
    printf("]\n");

    printf("  strides (bytes): [");
    for (int i = 0; i < t->ndim; ++i) {
        printf("%zu", t->strides[i]);
        if (i + 1 < t->ndim) {
            printf(", ");
        }
    }
    printf("]\n");

    printf("}\n");
}

bool tensor_compare(const tensor_t* a, const tensor_t* b) {
    if (a == NULL || b == NULL) {
        return false;
    }

    // Compare dtype
    if (a->dtype != b->dtype) {
        return false;
    }

    // Compare ndim
    if (a->ndim != b->ndim) {
        return false;
    }

    // Compare shape
    for (int i = 0; i < a->ndim; ++i) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }

    // Compare strides
    for (int i = 0; i < a->ndim; ++i) {
        if (a->strides[i] != b->strides[i]) {
            return false;
        }
    }

    // Compare size
    if (a->size != b->size) {
        return false;
    }

    // Compare bytes
    if (a->bytes != b->bytes) {
        return false;
    }

    // Compare data contents
    if (memcmp(a->data, b->data, a->bytes) != 0) {
        return false;
    }

    return true;
}
