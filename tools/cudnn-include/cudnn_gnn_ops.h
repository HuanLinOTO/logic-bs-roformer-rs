/*
 * Copyright 2014-2026 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

/*
 *  cudnn_gnn_ops : cuDNN's GNN (Graph Neural Network) operations.
 *
 *  Provides direct function-call APIs for GNN kernels (CSC graph aggregation,
 *  etc.) using NVRTC runtime compilation.
 */

#if !defined(CUDNN_GNN_OPS_H_)
#define CUDNN_GNN_OPS_H_

#pragma once

#include <cuda_runtime_api.h>

#include "cudnn_version.h"
#include "cudnn_ops.h"

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/* Aggregation operation for GNN CSC graph aggregation */
typedef enum {
    CUDNN_GNN_AGG_SUM  = 0,
    CUDNN_GNN_AGG_MEAN = 1,
    CUDNN_GNN_AGG_MAX  = 2,
    CUDNN_GNN_AGG_MIN  = 3,
} cudnnGnnAggOp_t;

/*
 * cudnnGnnCscGraph_t
 *
 * CSC-format graph descriptor. All pointer fields refer to device memory whose
 * element type matches idxType (CUDNN_DATA_INT32 or CUDNN_DATA_INT64).
 *
 * mapCscToCoo and mapRevToCoo are optional remapping arrays of length nIndices.
 * They are only needed when edge features are stored in a different order from
 * the CSC/reversed-CSC index arrays:
 *   mapCscToCoo[i] = position in the edge-feature table corresponding to
 *                    cscIndices[i].  NULL if edge features are already in
 *                    CSC index order.
 *   mapRevToCoo[i] = same mapping for the reversed graph (used by backward).
 *                    NULL if not needed.
 */
typedef struct {
    const void *cscOffsets;  /* length nDstNodes + 1 */
    const void *cscIndices;  /* length nIndices */
    const void *mapCscToCoo; /* optional; length nIndices; NULL if edge features are in CSC order */
    const void *mapRevToCoo; /* optional; length nIndices; NULL if not needed */
    int64_t nSrcNodes;
    int64_t nDstNodes;
    int64_t nIndices;
    cudnnDataType_t idxType; /* CUDNN_DATA_INT32 or CUDNN_DATA_INT64 */
} cudnnGnnCscGraph_t;

/*
 * cudnnGnnAggSimpleForward
 *
 * Performs neighborhood aggregation on a CSC-format graph using NVRTC-compiled
 * kernels (JIT-compiled on first call per (dataType, idxType), cached thereafter).
 * The JIT cache is process-global and thread-safe.
 *
 * For each destination node, aggregates features from its source neighbors
 * using the specified operation (sum, mean, max, min).
 *
 * Requires SM 8.0 (Ampere) or later. Returns CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH
 * on older devices.
 *
 * Parameters:
 *   stream          - CUDA stream for kernel launch (NULL for default stream)
 *   graph           - CSC graph descriptor; see cudnnGnnCscGraph_t
 *   nodeFeatures    - input node features, device memory, [nSrcNodes * nodeFeatDim].
 *                     NULL iff nodeFeatDim == 0 (pure edge aggregation).
 *   edgeFeatures    - input edge features, device memory, [nIndices * edgeFeatDim].
 *                     NULL iff edgeFeatDim == 0.
 *   concatFeatures  - per-destination-node features to pass through without aggregation,
 *                     device memory, [nDstNodes * concatFeatDim]. These are appended
 *                     as-is to each output row and are not aggregated over neighbors.
 *                     NULL iff concatFeatDim == 0.
 *   output          - output features, device memory,
 *                     [nDstNodes * (nodeFeatDim + edgeFeatDim + concatFeatDim)].
 *                     Layout per row: [aggregated_node | aggregated_edge | concat].
 *   outPositions    - argmax/argmin source-neighbor indices (MAX/MIN only), device memory,
 *                     [nDstNodes * (nodeFeatDim + edgeFeatDim)], integer type matching
 *                     graph.idxType. Must be non-NULL for MAX/MIN; pass NULL for SUM/MEAN.
 *                     Required by cudnnGnnAggSimpleBackward for MAX/MIN.
 *   nodeFeatDim     - feature dimension per node; 0 for pure edge aggregation.
 *                     At least one of nodeFeatDim and edgeFeatDim must be > 0.
 *   edgeFeatDim     - feature dimension per edge; 0 iff edgeFeatures is NULL.
 *   concatFeatDim   - pass-through feature dimension per destination node;
 *                     0 iff concatFeatures is NULL.
 *   dataType        - element type for all feature arrays. Supported: FLOAT, HALF, BFLOAT16.
 *   aggOp           - aggregation operation: SUM, MEAN, MAX, or MIN.
 *
 * Returns:
 *   CUDNN_STATUS_SUCCESS                      on success
 *   CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH  if the current device is below SM 8.0
 *   CUDNN_STATUS_BAD_PARAM                    if any pointer/dimension constraint is violated
 *   CUDNN_STATUS_INTERNAL_ERROR               if the CUDA device query fails unexpectedly
 */
cudnnStatus_t CUDNNWINAPI
cudnnGnnAggSimpleForward(cudaStream_t stream,
                         const cudnnGnnCscGraph_t *graph,
                         const void *nodeFeatures,
                         const void *edgeFeatures,
                         const void *concatFeatures,
                         void *output,
                         void *outPositions,
                         int nodeFeatDim,
                         int edgeFeatDim,
                         int concatFeatDim,
                         cudnnDataType_t dataType,
                         cudnnGnnAggOp_t aggOp);

/*
 * cudnnGnnAggSimpleBackward
 *
 * Computes gradients for cudnnGnnAggSimpleForward. For each destination node, reads
 * the upstream gradient and scatters contributions back to source neighbors via
 * atomic-add (non-deterministic across runs; order of atomics is not guaranteed).
 *
 * Requires SM 8.0 (Ampere) or later. Returns CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH
 * on older devices.
 *
 * Parameters:
 *   stream              - CUDA stream for kernel launch (NULL for default stream)
 *   graph               - CSC graph descriptor; must be identical to the forward call.
 *   gradOutput          - upstream gradient, device memory,
 *                         [nDstNodes * (nodeFeatDim + edgeFeatDim + concatFeatDim)].
 *                         Layout must match the forward output buffer.
 *   outPositions        - argmax/argmin indices written by the forward pass, device memory,
 *                         [nDstNodes * (nodeFeatDim + edgeFeatDim)], integer type matching
 *                         graph.idxType. Required for MAX/MIN; pass NULL for SUM/MEAN.
 *   gradNodeFeatures    - output: grad w.r.t. node features, device memory,
 *                         [nSrcNodes * nodeFeatDim]. NULL iff nodeFeatDim == 0.
 *                         Must be zero-initialised by the caller (kernel uses atomic-add).
 *   gradEdgeFeatures    - output: grad w.r.t. edge features, device memory,
 *                         [nIndices * edgeFeatDim]. NULL iff edgeFeatDim == 0.
 *                         Must be zero-initialised by the caller (kernel uses atomic-add).
 *   gradConcatFeatures  - output: grad w.r.t. pass-through concat features, device memory,
 *                         [nDstNodes * concatFeatDim]. NULL iff concatFeatDim == 0.
 *                         Written directly (one row per dst node, no atomics).
 *   nodeFeatDim         - feature dimension per node; must match the forward call.
 *   edgeFeatDim         - feature dimension per edge; must match the forward call.
 *   concatFeatDim       - pass-through feature dimension; must match the forward call.
 *   dataType            - element type; must match the forward call.
 *   aggOp               - aggregation operation; must match the forward call.
 *
 * Returns:
 *   CUDNN_STATUS_SUCCESS                      on success
 *   CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH  if the current device is below SM 8.0
 *   CUDNN_STATUS_BAD_PARAM                    if any pointer/dimension constraint is violated
 *   CUDNN_STATUS_INTERNAL_ERROR               if the CUDA device query fails unexpectedly
 */
cudnnStatus_t CUDNNWINAPI
cudnnGnnAggSimpleBackward(cudaStream_t stream,
                          const cudnnGnnCscGraph_t *graph,
                          const void *gradOutput,
                          const void *outPositions,
                          void *gradNodeFeatures,
                          void *gradEdgeFeatures,
                          void *gradConcatFeatures,
                          int nodeFeatDim,
                          int edgeFeatDim,
                          int concatFeatDim,
                          cudnnDataType_t dataType,
                          cudnnGnnAggOp_t aggOp);

#if defined(__cplusplus)
}
#endif

#endif /* CUDNN_GNN_OPS_H_ */
