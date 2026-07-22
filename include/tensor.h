#ifndef PHOBOSML_TENSOR_H
#define PHOBOSML_TENSOR_H

#include <stdbool.h>  // bool
#include <stddef.h>   // size_t
#include <stdint.h>   // uint32_t, uint64_t

/**
 * @file tensor.h
 * @brief Core tensor structure and API for PHOBOS-ML.
 *
 * This header defines the fundamental tensor type used throughout the
 * PHOBOS-ML engine. It includes metadata such as shape, strides, dtype,
 * and memory ownership flags. All operations, autograd components, and
 * model layers depend on this interface.
 *
 * The header is designed to compile standalone and serve as a stable
 * public API for C, Python (via FFI), and other language bindings.
 */

/* ============================================================
   Supported data types
   ============================================================ */

/**
 * @enum dtype_t
 * @brief Enumerates supported tensor element types.
 */
typedef enum {
    DTYPE_FLOAT32, /**< 32-bit floating point */
    DTYPE_FLOAT64, /**< 64-bit floating point */
    DTYPE_INT32,   /**< 32-bit integer */
    DTYPE_INT64    /**< 64-bit integer */
} dtype_t;

/* ============================================================
   Tensor structure
   ============================================================ */

/**
 * @struct tensor_t
 * @brief Represents an n-dimensional array with metadata.
 *
 * A tensor contains:
 * - A raw memory pointer (`data`)
 * - A data type (`dtype`)
 * - Dimensionality (`ndim`)
 * - Shape array (`shape`)
 * - Strides array (`strides`)
 * - Total number of elements (`size`)
 * - Total memory footprint (`bytes`)
 * - Ownership flag (`owns_data`)
 *
 * Strides are expressed in **bytes**, not elements.
 */
typedef struct {
    void* data;      /**< Pointer to raw memory */
    dtype_t dtype;   /**< Element type */
    int ndim;        /**< Number of dimensions */
    int* shape;      /**< Array of dimension sizes */
    size_t* strides; /**< Array of strides (in bytes) */
    size_t size;     /**< Total number of elements */
    size_t bytes;    /**< Total memory size in bytes */
    bool owns_data;  /**< True if tensor owns its memory */
} tensor_t;

/* ============================================================
   Creation / destruction
   ============================================================ */

/**
 * @brief Creates a new tensor and allocates memory.
 *
 * Computes strides, total size, and memory footprint. Allocates
 * contiguous memory for the tensor elements.
 *
 * @param dtype Element type.
 * @param ndim Number of dimensions.
 * @param shape Array of dimension sizes.
 * @return Pointer to a newly allocated tensor, or NULL on failure.
 */
tensor_t* tensor_create(dtype_t dtype, int ndim, const int* shape);

/**
 * @brief Creates a tensor that views external memory.
 *
 * Does NOT allocate memory. Useful for wrapping buffers from Python,
 * Java, or other languages.
 *
 * @param data External memory pointer.
 * @param dtype Element type.
 * @param ndim Number of dimensions.
 * @param shape Array of dimension sizes.
 * @return Pointer to a tensor view, or NULL on failure.
 */
tensor_t* tensor_from_data(void* data, dtype_t dtype, int ndim, const int* shape);

/**
 * @brief Frees a tensor and its memory if owned.
 *
 * If `owns_data` is false, only metadata is freed.
 *
 * @param t Tensor to free.
 */
void tensor_free(tensor_t* t);

/* ============================================================
   Utility functions
   ============================================================ */

/**
 * @brief Returns the size in bytes of a single element of the given dtype.
 *
 * @param dtype Element type.
 * @return Size in bytes.
 */
size_t tensor_dtype_size(dtype_t dtype);

/**
 * @brief Checks whether the tensor is stored contiguously in memory.
 *
 * @param t Tensor to inspect.
 * @return True if contiguous, false otherwise.
 */
bool tensor_is_contiguous(const tensor_t* t);

/**
 * @brief Prints tensor metadata for debugging.
 *
 * @param t Tensor to print.
 */
void tensor_print(const tensor_t* t);

#endif  // PHOBOSML_TENSOR_H

/**
 * @brief Compares two tensors for equality (metadata + data).
 *
 * Does not print anything. Intended for tests and debugging.
 *
 * @param a First tensor.
 * @param b Second tensor.
 * @return true if tensors are identical, false otherwise.
 */
bool tensor_compare(const tensor_t* a, const tensor_t* b);
