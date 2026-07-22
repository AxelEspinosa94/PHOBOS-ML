#include <stdio.h>

#include "tensor.h"


int main(void) {
    int shape[1] = {5};
    tensor_t* t = tensor_create(DTYPE_INT32, 1, shape);

    if (!t) {
        printf("FAIL: tensor_create returned NULL\n");
        return 1;
    }

    tensor_free(t);

    printf("PASS: tensor_free\n");
    return 0;
}
