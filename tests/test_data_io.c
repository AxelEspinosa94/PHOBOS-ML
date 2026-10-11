#include <math.h>
#include <stdio.h>

#include "data_io.h"
#include "tensor.h"

static int almost_equal(float a, float b) {
    return fabsf(a - b) < 1e-6f;
}

int main(void) {
    int shape[2] = {2, 3};

    tensor_t* A =
        tensor_create(
            DTYPE_FLOAT32,
            2,
            shape);

    if (!A) {
        printf("FAIL: tensor_create\n");
        return 1;
    }

    float* ad = (float*)A->data;

    ad[0] = 1.0f;
    ad[1] = 2.0f;
    ad[2] = 3.0f;

    ad[3] = 4.0f;
    ad[4] = 5.0f;
    ad[5] = 6.0f;

    int err =
        tensor_save_csv(
            "test_data.csv",
            A);

    if (err != 0) {
        printf("FAIL: save_csv\n");
        tensor_free(A);
        return 1;
    }

    tensor_t* B =
        tensor_load_csv(
            "test_data.csv",
            0);

    if (!B) {
        printf("FAIL: load_csv\n");
        tensor_free(A);
        return 1;
    }

    if (B->shape[0] != 2) {
        printf("FAIL: rows mismatch\n");
        tensor_free(A);
        tensor_free(B);
        return 1;
    }

    if (B->shape[1] != 3) {
        printf("FAIL: cols mismatch\n");
        tensor_free(A);
        tensor_free(B);
        return 1;
    }

    float* bd = (float*)B->data;

    for (size_t i = 0; i < A->size; ++i) {
        if (!almost_equal(ad[i], bd[i])) {
            printf(
                "FAIL: value mismatch at index %zu "
                "(expected %.6f, got %.6f)\n",
                i,
                ad[i],
                bd[i]);

            tensor_free(A);
            tensor_free(B);
            return 1;
        }
    }

    tensor_free(A);
    tensor_free(B);

    printf("PASS: data_io\n");

    return 0;
}