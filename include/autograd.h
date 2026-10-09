#ifndef AUTOGRAD_H
#define AUTOGRAD_H

/**
 * @struct Value
 * @brief Represents an n-dimensional array with metadata.
 *
 * A Value contains:
 * - data: the value of the node
 * - grad: the gradient of the node
 * - left: the left child node
 * - right: the right child node
 * - backward: the backward function for gradient computation
 *
 * Strides are expressed in **bytes**, not elements.
 */
typedef struct Value Value;

struct Value {
    float data;
    float grad;

    Value* left;
    Value* right;

    void (*backward)(Value*);
};

Value* value_create(float data);

Value* value_add(Value* a, Value* b);
Value* value_mul(Value* a, Value* b);

Value* value_relu(Value* x);
Value* value_sigmoid(Value* x);

void value_backward(Value* root);

void value_free(Value* v);

#endif