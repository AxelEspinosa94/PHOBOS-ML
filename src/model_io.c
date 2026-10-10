#include "model_io.h"

#include <stdio.h>
#include <string.h>

#define PHOBOS_MAGIC "PHOBOSML"

int model_save(
    const char* path,
    const tensor_t* W,
    float b) {
    if (!path || !W)
        return -1;

    FILE* fp = fopen(path, "wb");

    if (!fp)
        return -2;

    fwrite(PHOBOS_MAGIC,
           sizeof(char),
           strlen(PHOBOS_MAGIC),
           fp);

    fwrite(&W->ndim,
           sizeof(int),
           1,
           fp);

    fwrite(W->shape,
           sizeof(int),
           W->ndim,
           fp);

    fwrite(&W->size,
           sizeof(size_t),
           1,
           fp);

    fwrite(W->data,
           sizeof(float),
           W->size,
           fp);

    fwrite(&b,
           sizeof(float),
           1,
           fp);

    fclose(fp);

    return 0;
}

int model_load(
    const char* path,
    tensor_t* W,
    float* b) {
    if (!path || !W || !b)
        return -1;

    FILE* fp = fopen(path, "rb");

    if (!fp)
        return -2;

    char magic[9] = {0};

    fread(magic,
          sizeof(char),
          8,
          fp);

    if (strcmp(magic, PHOBOS_MAGIC) != 0) {
        fclose(fp);
        return -3;
    }

    int ndim = 0;

    fread(&ndim,
          sizeof(int),
          1,
          fp);

    if (ndim != W->ndim) {
        fclose(fp);
        return -4;
    }

    int shape[ndim];

    fread(shape,
          sizeof(int),
          ndim,
          fp);

    for (int i = 0; i < ndim; ++i) {
        if (shape[i] != W->shape[i]) {
            fclose(fp);
            return -5;
        }
    }

    size_t size = 0;

    fread(&size,
          sizeof(size_t),
          1,
          fp);

    if (size != W->size) {
        fclose(fp);
        return -6;
    }

    fread(W->data,
          sizeof(float),
          W->size,
          fp);

    fread(b,
          sizeof(float),
          1,
          fp);

    fclose(fp);

    return 0;
}