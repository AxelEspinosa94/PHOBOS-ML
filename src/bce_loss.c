#include <math.h>
#include "tensor.h"

float tensor_bce_loss(const tensor_t* y, const tensor_t* p) {
    if (!y || !p) return 0.0f;
    if (y->dtype != DTYPE_FLOAT32 || p->dtype != DTYPE_FLOAT32) return 0.0f;
    if (y->size != p->size) return 0.0f;

    float* yd = (float*)y->data;
    float* pd = (float*)p->data;

    const float eps = 1e-7f;  // numerical stability
    float loss_sum = 0.0f;

    for (size_t i = 0; i < y->size; ++i) {
        float yi = yd[i];
        float pi = pd[i];

        // Clamp predictions to avoid log(0)
        if (pi < eps) pi = eps;
        if (pi > 1.0f - eps) pi = 1.0f - eps;

        float term = yi * logf(pi) + (1.0f - yi) * logf(1.0f - pi);
        loss_sum += term;
    }

    return -loss_sum / (float)y->size;
}
