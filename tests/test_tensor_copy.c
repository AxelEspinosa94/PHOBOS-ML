#include "tensor.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    int shape[2] = {2, 2};
    float data[4] = {1, 2, 3, 4};

    tensor_t* t = tensor_from_data(data, DTYPE_FLOAT32, 2, shape);

    if (!t) {
        printf("FAIL: tensor_from_data returned NULL\n");
        return 1;
    }

    if (memcmp(t->data, data, sizeof(data)) != 0) {
        printf("FAIL: tensor_from_data data mismatch\n");
        return 1;
    }

    printf("PASS: tensor_from_data\n");

    tensor_free(t);
    return 0;
}
