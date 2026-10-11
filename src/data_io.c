#include "data_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CSV_MAX_LINE 4096

static int count_columns(char* line) {
    int cols = 1;

    for (char* p = line; *p; ++p) {
        if (*p == ',')
            cols++;
    }

    return cols;
}

tensor_t* tensor_load_csv(
    const char* path,
    int skip_header) {
    if (!path)
        return NULL;

    FILE* fp = fopen(path, "r");

    if (!fp)
        return NULL;

    char line[CSV_MAX_LINE];

    int rows = 0;
    int cols = 0;

    if (skip_header) {
        if (fgets(line, sizeof(line), fp) == NULL) {
            fclose(fp);
            return NULL;
        }
    }

    while (fgets(line, sizeof(line), fp)) {
        if (rows == 0)
            cols = count_columns(line);

        if (count_columns(line) != cols) {
            fclose(fp);
            return NULL;
        }

        rows++;
    }

    rewind(fp);

    if (skip_header) {
        if (fgets(line, sizeof(line), fp) == NULL) {
            fclose(fp);
            return NULL;
        }
    }

    int shape[2] = {rows, cols};

    tensor_t* tensor =
        tensor_create(
            DTYPE_FLOAT32,
            2,
            shape);

    if (!tensor) {
        fclose(fp);
        return NULL;
    }

    float* data = (float*)tensor->data;

    int r = 0;

    while (fgets(line, sizeof(line), fp)) {
        int c = 0;

        char* token =
            strtok(line, ",");

        while (token) {
            char* endptr;

            float value =
                strtof(
                    token,
                    &endptr);

            if (endptr == token ||
                (*endptr != '\0' &&
                 *endptr != '\n' &&
                 *endptr != '\r')) {
                tensor_free(tensor);
                fclose(fp);
                return NULL;
            }

            data[r * cols + c] = value;

            c++;

            token =
                strtok(NULL, ",");
        }

        r++;
    }

    fclose(fp);

    return tensor;
}

int tensor_save_csv(
    const char* path,
    const tensor_t* tensor) {
    if (!path || !tensor)
        return -1;

    if (tensor->ndim != 2)
        return -2;

    FILE* fp = fopen(path, "w");

    if (!fp)
        return -3;

    int rows = tensor->shape[0];
    int cols = tensor->shape[1];

    float* data =
        (float*)tensor->data;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            fprintf(
                fp,
                "%f",
                data[r * cols + c]);

            if (c < cols - 1)
                fprintf(fp, ",");
        }

        fprintf(fp, "\n");
    }

    fclose(fp);

    return 0;
}