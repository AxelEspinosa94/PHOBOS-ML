#include <math.h>
#include <stdio.h>

#include "tensor.h"

/* ============================
   Helpers
   ============================ */

static int almost_equal(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

/* ============================
   ReLU tests
   ============================ */

int test_relu_basic(void) {
    int shape[1] = {5};
    tensor_t* A = tensor_create(DTYPE_FLOAT32, 1, shape);

    float* ad = (float*)A->data;
    ad[0] = -3.0f;
    ad[1] = -1.0f;
    ad[2] = 0.0f;
    ad[3] = 2.0f;
    ad[4] = 5.0f;

    tensor_t* C = tensor_relu(A);
    if (!C) return 1;

    float* cd = (float*)C->data;

    if (cd[0] != 0.0f) return 1;
    if (cd[1] != 0.0f) return 1;
    if (cd[2] != 0.0f) return 1;
    if (cd[3] != 2.0f) return 1;
    if (cd[4] != 5.0f) return 1;

    tensor_free(A);
    tensor_free(C);
    return 0;
}

/* ============================
   Sigmoid tests
   ============================ */

int test_sigmoid_basic(void) {
    int shape[1] = {3};
    tensor_t* A = tensor_create(DTYPE_FLOAT32, 1, shape);

    float* ad = (float*)A->data;
    ad[0] = 0.0f;
    ad[1] = 10.0f;
    ad[2] = -10.0f;

    tensor_t* C = tensor_sigmoid(A);
    if (!C) return 1;

    float* cd = (float*)C->data;

    if (!almost_equal(cd[0], 0.5f, 1e-6f)) return 1;
    if (!almost_equal(cd[1], 0.9999546f, 1e-6f)) return 1;
    if (!almost_equal(cd[2], 0.0000454f, 1e-6f)) return 1;

    tensor_free(A);
    tensor_free(C);
    return 0;
}

/* ============================
   Softmax tests
   ============================ */

int test_softmax_vector(void) {
    int shape[1] = {3};
    tensor_t* A = tensor_create(DTYPE_FLOAT32, 1, shape);

    float* ad = (float*)A->data;
    ad[0] = 1.0f;
    ad[1] = 2.0f;
    ad[2] = 3.0f;

    tensor_t* C = tensor_softmax(A);
    if (!C) return 1;

    float* cd = (float*)C->data;

    if (!almost_equal(cd[0], 0.09003057f, 1e-5f)) return 1;
    if (!almost_equal(cd[1], 0.24472847f, 1e-5f)) return 1;
    if (!almost_equal(cd[2], 0.66524096f, 1e-5f)) return 1;

    tensor_free(A);
    tensor_free(C);
    return 0;
}

int test_softmax_batch(void) {
    int shape[2] = {2, 3};
    tensor_t* A = tensor_create(DTYPE_FLOAT32, 2, shape);

    float* ad = (float*)A->data;
    ad[0] = 1.0f;
    ad[1] = 2.0f;
    ad[2] = 3.0f;
    ad[3] = 4.0f;
    ad[4] = 5.0f;
    ad[5] = 6.0f;

    tensor_t* C = tensor_softmax(A);
    if (!C) return 1;

    float* cd = (float*)C->data;

    /* First row */
    if (!almost_equal(cd[0], 0.09003057f, 1e-5f)) return 1;
    if (!almost_equal(cd[1], 0.24472847f, 1e-5f)) return 1;
    if (!almost_equal(cd[2], 0.66524096f, 1e-5f)) return 1;

    /* Second row */
    if (!almost_equal(cd[3], 0.09003057f, 1e-5f)) return 1;
    if (!almost_equal(cd[4], 0.24472847f, 1e-5f)) return 1;
    if (!almost_equal(cd[5], 0.66524096f, 1e-5f)) return 1;

    tensor_free(A);
    tensor_free(C);
    return 0;
}

/* ============================
   Main runner
   ============================ */

int main(void) {
    if (test_relu_basic()) {
        printf("FAIL: ReLU\n");
        return 1;
    }

    if (test_sigmoid_basic()) {
        printf("FAIL: Sigmoid\n");
        return 1;
    }

    if (test_softmax_vector()) {
        printf("FAIL: Softmax vector\n");
        return 1;
    }

    if (test_softmax_batch()) {
        printf("FAIL: Softmax batch\n");
        return 1;
    }

    printf("PASS: activations\n");
    return 0;
}
