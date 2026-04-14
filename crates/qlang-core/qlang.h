/*
 * qlang.h — C FFI for the QLANG graph engine
 *
 * Link against the qlang-core shared/static library built by Cargo.
 *
 * All `QlangGraph*` pointers are opaque — never dereference them from C.
 */

#ifndef QLANG_H
#define QLANG_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a QLANG graph. */
typedef struct Graph QlangGraph;

/*
 * Dtype encoding (pass as uint32_t):
 *   0 = F16
 *   1 = F32
 *   2 = F64
 *   3 = I8
 *   4 = I16
 *   5 = I32
 *   6 = I64
 *   7 = Bool
 *   8 = Ternary
 */

/**
 * Create a new empty graph.
 *
 * @param name  Null-terminated UTF-8 graph name.
 * @return      Pointer to the new graph, or NULL on error.
 *              Must be freed with qlang_graph_free().
 */
QlangGraph *qlang_graph_new(const char *name);

/**
 * Free a graph created by qlang_graph_new().
 * Passing NULL is a safe no-op.
 */
void qlang_graph_free(QlangGraph *graph);

/**
 * Add an input node to the graph.
 *
 * @param graph     Graph handle.
 * @param name      Null-terminated input name.
 * @param dtype     Data type (see encoding above).
 * @param shape_ptr Pointer to array of dimension sizes.
 * @param shape_len Number of dimensions.
 * @return          Node ID, or UINT32_MAX on error.
 */
uint32_t qlang_graph_add_input(QlangGraph *graph,
                               const char *name,
                               uint32_t dtype,
                               const size_t *shape_ptr,
                               size_t shape_len);

/**
 * Add an operation node and wire it to inputs.
 *
 * @param graph    Graph handle.
 * @param op_name  Null-terminated op name (e.g. "add", "relu", "matmul").
 * @param input_a  Node ID of first input (UINT32_MAX if unused).
 * @param input_b  Node ID of second input (UINT32_MAX if unused).
 * @return         Node ID, or UINT32_MAX on error.
 */
uint32_t qlang_graph_add_op(QlangGraph *graph,
                            const char *op_name,
                            uint32_t input_a,
                            uint32_t input_b);

/**
 * Add an output node to the graph.
 *
 * @param graph   Graph handle.
 * @param name    Null-terminated output name.
 * @param source  Node ID whose output feeds this output node.
 * @return        Node ID, or UINT32_MAX on error.
 */
uint32_t qlang_graph_add_output(QlangGraph *graph,
                                const char *name,
                                uint32_t source);

/**
 * Execute a graph (stub — returns -1 until runtime is wired).
 *
 * @param graph        Graph handle.
 * @param input_names  Array of null-terminated input names.
 * @param input_data   Array of float pointers, one per input.
 * @param n_inputs     Number of inputs.
 * @param output_data  Caller-allocated buffer for output values.
 * @param output_len   Length of the output buffer.
 * @return             0 on success, negative on error.
 */
int32_t qlang_graph_execute(QlangGraph *graph,
                            const char *const *input_names,
                            const float *const *input_data,
                            size_t n_inputs,
                            float *output_data,
                            size_t output_len);

/**
 * Serialize the graph to a JSON string.
 *
 * @param graph  Graph handle.
 * @return       Heap-allocated JSON string, or NULL on error.
 *               Must be freed with qlang_free_string().
 */
char *qlang_graph_to_json(QlangGraph *graph);

/**
 * Free a string returned by qlang_graph_to_json().
 * Passing NULL is a safe no-op.
 */
void qlang_free_string(char *s);

/**
 * Return the number of nodes in the graph.
 *
 * @param graph  Graph handle (NULL returns 0).
 * @return       Node count.
 */
uint32_t qlang_graph_num_nodes(QlangGraph *graph);

/**
 * Verify graph structural and type correctness.
 *
 * @param graph  Graph handle.
 * @return       0 if verification passes, -1 on NULL,
 *               positive integer = number of failures.
 */
int32_t qlang_graph_verify(QlangGraph *graph);

/**
 * Return the QLANG version string.
 * The returned pointer is static and must NOT be freed.
 */
const char *qlang_version(void);

#ifdef __cplusplus
}
#endif

#endif /* QLANG_H */
