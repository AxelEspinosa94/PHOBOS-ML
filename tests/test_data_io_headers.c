// tests/test_data_io_headers.c

#include <math.h>
#include <stdio.h>

#include "data_io.h"
#include "tensor.h"

static int almost_equal(float a, float b) {
    return fabsf(a - b) < 1e-6f;
}

int main(void) {
    /* ==================================
       Create CSV with headers manually
       ================================== */

    FILE* fp = fopen("test_headers.csv", "w");

    if (!fp) {
        printf("FAIL: create csv\n");
        return 1;
    }

    fprintf(fp, "x,y\n");
    fprintf(fp, "1,2\n");
    fprintf(fp, "3,4\n");

    fclose(fp);

    /* ==================================
       Load CSV skipping header
       ================================== */

    tensor_t* T =
        tensor_load_csv(
            "test_headers.csv",
            1);

    if (!T) {
        printf("FAIL: load with header\n");
        return 1;
    }

    if (T->shape[0] != 2) {
        printf("FAIL: row count\n");
        return 1;
    }

    if (T->shape[1] != 2) {
        printf("FAIL: column count\n");
        return 1;
    }

    float* td = (float*)T->data;

    if (!almost_equal(td[0], 1.0f)) return 1;
    if (!almost_equal(td[1], 2.0f)) return 1;
    if (!almost_equal(td[2], 3.0f)) return 1;
    if (!almost_equal(td[3], 4.0f)) return 1;

    tensor_free(T);

    /* ==================================
       Verify failure when header is not skipped
       ================================== */

    tensor_t* Invalid =
        tensor_load_csv(
            "test_headers.csv",
            0);

    /*
       Depending on how your parser behaves,
       you may decide either:

       A) return NULL
       B) throw parsing error

       I recommend NULL.
    */

    if (Invalid != NULL) {
        printf(
            "FAIL: header should not be parsed as numeric data\n");

        tensor_free(Invalid);

        return 1;
    }

    printf("PASS: data_io_headers\n");

    return 0;
}