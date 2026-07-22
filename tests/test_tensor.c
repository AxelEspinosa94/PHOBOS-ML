#include <stdio.h>

#include "tensor.h"

int main(void) {
    printf("=== PHOBOS-ML Tensor Test ===\n");

    // Shape: [2, 3]
    int shape[2] = {2, 3};

    // Create tensor
    tensor_t* t = tensor_create(DTYPE_FLOAT32, 2, shape);
    if (t == NULL) {
        printf("ERROR: tensor_create returned NULL\n");
        return 1;
    }

    printf("Tensor created successfully.\n");
    tensor_print(t);

    // Check contiguity
    if (tensor_is_contiguous(t)) {
        printf("Tensor is contiguous.\n");
    } else {
        printf("Tensor is NOT contiguous.\n");
    }

    // Fill tensor with some values
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; ++i) {
        data[i] = (float)i * 1.5f;
    }

    printf("Data filled. First element: %.2f\n", data[0]);
    printf("Last element: %.2f\n", data[t->size - 1]);

    printf("=== Testing tensor_compare ===\n");

    tensor_t* a = tensor_create(DTYPE_FLOAT32, 2, shape);
    tensor_t* b = tensor_create(DTYPE_FLOAT32, 2, shape);

    float* ad = (float*)a->data;
    float* bd = (float*)b->data;

    for (size_t i = 0; i < a->size; ++i) {
        ad[i] = (float)i;
        bd[i] = (float)i;
    }

    if (tensor_compare(a, b)) {
        printf("tensor_compare: OK (tensors are equal)\n");
    } else {
        printf("tensor_compare: FAIL (tensors should be equal)\n");
    }

    bd[0] = 999.0f;

    if (!tensor_compare(a, b)) {
        printf("tensor_compare: OK (difference detected)\n");
    } else {
        printf("tensor_compare: FAIL (should detect difference)\n");
    }

    tensor_free(a);
    tensor_free(b);

    // Free tensor
    tensor_free(t);
    printf("Tensor freed successfully.\n");

    return 0;
}
