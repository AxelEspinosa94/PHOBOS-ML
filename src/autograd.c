#include "autograd.h"

#include <math.h>
#include <stdlib.h>

static void backward_add(Value* v) {
    v->left->grad += v->grad;
    v->right->grad += v->grad;
}

static void backward_mul(Value* v) {
    v->left->grad += v->right->data * v->grad;
    v->right->grad += v->left->data * v->grad;
}

static void backward_relu(Value* v) {
    if (v->left->data > 0.0f)
        v->left->grad += v->grad;
}

static void backward_sigmoid(Value* v) {
    float s = v->data;
    v->left->grad += (s * (1.0f - s)) * v->grad;
}

Value* value_create(float data) {
    Value* v = (Value*)malloc(sizeof(Value));

    if (!v)
        return NULL;

    v->data = data;
    v->grad = 0.0f;

    v->left = NULL;
    v->right = NULL;

    v->backward = NULL;

    return v;
}

Value* value_add(Value* a, Value* b) {
    Value* out = value_create(a->data + b->data);

    if (!out)
        return NULL;

    out->left = a;
    out->right = b;

    out->backward = backward_add;

    return out;
}

Value* value_mul(Value* a, Value* b) {
    Value* out = value_create(a->data * b->data);

    if (!out)
        return NULL;

    out->left = a;
    out->right = b;

    out->backward = backward_mul;

    return out;
}

Value* value_relu(Value* x) {
    Value* out =
        value_create(
            x->data > 0.0f ? x->data : 0.0f);

    if (!out)
        return NULL;

    out->left = x;

    out->backward = backward_relu;

    return out;
}

Value* value_sigmoid(Value* x) {
    float s =
        1.0f /
        (1.0f + expf(-x->data));

    Value* out = value_create(s);

    if (!out)
        return NULL;

    out->left = x;

    out->backward = backward_sigmoid;

    return out;
}

void value_backward(Value* root) {
    if (!root)
        return;

    root->grad = 1.0f;

    if (root->backward)
        root->backward(root);

    if (root->left && root->left->backward)
        root->left->backward(root->left);

    if (root->right && root->right->backward)
        root->right->backward(root->right);
}

void value_free(Value* v) {
    if (!v)
        return;

    free(v);
}