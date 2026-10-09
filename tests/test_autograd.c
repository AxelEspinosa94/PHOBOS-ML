#include <math.h>
#include <stdio.h>

#include "autograd.h"

static int almost_equal(float a,
                        float b,
                        float eps) {
    return fabsf(a - b) < eps;
}

static int test_add(void) {
    Value* a = value_create(2.0f);
    Value* b = value_create(3.0f);

    Value* c = value_add(a, b);

    value_backward(c);

    int ok =
        almost_equal(a->grad, 1.0f, 1e-6f) &&
        almost_equal(b->grad, 1.0f, 1e-6f);

    value_free(c);
    value_free(a);
    value_free(b);

    return ok ? 0 : 1;
}

static int test_mul(void) {
    Value* a = value_create(2.0f);
    Value* b = value_create(3.0f);

    Value* c = value_mul(a, b);

    value_backward(c);

    int ok =
        almost_equal(a->grad, 3.0f, 1e-6f) &&
        almost_equal(b->grad, 2.0f, 1e-6f);

    value_free(c);
    value_free(a);
    value_free(b);

    return ok ? 0 : 1;
}

static int test_relu(void) {
    Value* x = value_create(-1.0f);

    Value* y = value_relu(x);

    value_backward(y);

    int ok =
        almost_equal(y->data, 0.0f, 1e-6f) &&
        almost_equal(x->grad, 0.0f, 1e-6f);

    value_free(y);
    value_free(x);

    return ok ? 0 : 1;
}

static int test_sigmoid(void) {
    Value* x = value_create(0.0f);

    Value* y = value_sigmoid(x);

    value_backward(y);

    int ok =
        almost_equal(y->data, 0.5f, 1e-6f) &&
        almost_equal(x->grad, 0.25f, 1e-6f);

    value_free(y);
    value_free(x);

    return ok ? 0 : 1;
}

int main(void) {
    if (test_add()) {
        printf("FAIL: autograd add\n");
        return 1;
    }

    if (test_mul()) {
        printf("FAIL: autograd mul\n");
        return 1;
    }

    if (test_relu()) {
        printf("FAIL: autograd relu\n");
        return 1;
    }

    if (test_sigmoid()) {
        printf("FAIL: autograd sigmoid\n");
        return 1;
    }

    printf("PASS: autograd\n");

    return 0;
}