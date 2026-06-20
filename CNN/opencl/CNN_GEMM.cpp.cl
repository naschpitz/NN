#ifndef CNN_GEMM_CPP_CL
#define CNN_GEMM_CPP_CL

// Tiled General Matrix Multiply (GEMM) kernels for im2col-based convolution.
//
// WHY GEMM?
// Direct convolution (one work-item per output element, nested loops over filter×channels)
// has poor memory access patterns: each work-item reads scattered, non-contiguous memory
// locations with no data reuse between neighboring work-items. This wastes GPU memory
// bandwidth, which is the primary bottleneck.
//
// The im2col + GEMM approach reshapes convolution into a matrix multiplication:
//   Output = Filters × im2col(Input) + Bias
// Matrix multiplication has regular, predictable access patterns that GPUs excel at.
// By tiling the computation into TILE_SIZE × TILE_SIZE blocks loaded into fast local
// (shared) memory, each element is loaded once from global memory and reused TILE_SIZE
// times, reducing global memory traffic by ~TILE_SIZE×.
//
// HOW IT WORKS:
// Each work-group computes one TILE×TILE block of the output matrix C.
// The K dimension (shared between A and B) is processed in tiles:
//   1. All work-items collaboratively load one tile of A and one tile of B into local memory
//   2. barrier() ensures the tile is fully loaded
//   3. Each work-item computes a partial dot product using the local tiles
//   4. barrier() ensures the tile is fully consumed before loading the next one
//   5. After all K-tiles, the accumulated result is written to global memory
//
// Four variants handle the different matrix layouts needed for forward and backward passes:
//   gemm_conv          — C = A × B + bias  (forward: Output = Filters × im2col + Bias)
//   gemm_dInput        — C = A^T × B       (backward: dInput_cols = Filters^T × dOut)
//   gemm_dFilters      — C = A × B^T       (backward: dFilters = dOut × im2col^T)
//   gemm_dFilters_kpar — K-parallel variant of gemm_dFilters for small-output / large-K shapes
//
// Depends on CNN_Defines.hpp.cl for TYPE and TILE_SIZE.

//===================================================================================================================//

// Tiled GEMM: C = A × B + bias
// A: (M, K) at offsetA, B: (K, N) at offsetB, C: (M, N) at offsetC
// bias: vector of length M at offsetBias, added per-row
// 2D dispatch: global (ceil(N/TILE_SIZE)*TILE_SIZE, ceil(M/TILE_SIZE)*TILE_SIZE)
//              local  (TILE_SIZE, TILE_SIZE)
kernel void gemm_conv(global TYPE* A, global TYPE* B, global TYPE* C, global TYPE* bias, ulong offsetA, ulong offsetB,
                 ulong offsetC, ulong offsetBias, ulong M, ulong N, ulong K)
{
  uint localRow = get_local_id(1);
  uint localCol = get_local_id(0);
  uint globalRow = get_group_id(1) * TILE_SIZE + localRow;
  uint globalCol = get_group_id(0) * TILE_SIZE + localCol;

  local TYPE tileA[TILE_SIZE][TILE_SIZE];
  local TYPE tileB[TILE_SIZE][TILE_SIZE];

  TYPE acc = (TYPE)0;

  uint numTilesK = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (uint t = 0; t < numTilesK; t++) {
    uint aCol = t * TILE_SIZE + localCol;
    uint bRow = t * TILE_SIZE + localRow;

    // Load A[globalRow][aCol]
    if (globalRow < M && aCol < K)
      tileA[localRow][localCol] = A[offsetA + globalRow * K + aCol];
    else
      tileA[localRow][localCol] = (TYPE)0;

    // Load B[bRow][globalCol]
    if (bRow < K && globalCol < N)
      tileB[localRow][localCol] = B[offsetB + bRow * N + globalCol];
    else
      tileB[localRow][localCol] = (TYPE)0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < TILE_SIZE; k++)
      acc += tileA[localRow][k] * tileB[k][localCol];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (globalRow < M && globalCol < N)
    C[offsetC + globalRow * N + globalCol] = acc + bias[offsetBias + globalRow];
}

//===================================================================================================================//

// Tiled GEMM: C = A^T × B (no bias)
// A stored as (K, M), accessed as (M, K) via transposed indexing
// B: (K, N), C: (M, N)
kernel void gemm_dInput(global TYPE* A, global TYPE* B, global TYPE* C, ulong offsetA, ulong offsetB, ulong offsetC,
                        ulong M, ulong N, ulong K)
{
  uint localRow = get_local_id(1);
  uint localCol = get_local_id(0);
  uint globalRow = get_group_id(1) * TILE_SIZE + localRow;
  uint globalCol = get_group_id(0) * TILE_SIZE + localCol;

  local TYPE tileA[TILE_SIZE][TILE_SIZE];
  local TYPE tileB[TILE_SIZE][TILE_SIZE];

  TYPE acc = (TYPE)0;

  uint numTilesK = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (uint t = 0; t < numTilesK; t++) {
    uint aCol = t * TILE_SIZE + localCol;
    uint bRow = t * TILE_SIZE + localRow;

    // Load A^T[globalRow][aCol] = A[aCol][globalRow]
    if (globalRow < M && aCol < K)
      tileA[localRow][localCol] = A[offsetA + aCol * M + globalRow];
    else
      tileA[localRow][localCol] = (TYPE)0;

    // Load B[bRow][globalCol]
    if (bRow < K && globalCol < N)
      tileB[localRow][localCol] = B[offsetB + bRow * N + globalCol];
    else
      tileB[localRow][localCol] = (TYPE)0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < TILE_SIZE; k++)
      acc += tileA[localRow][k] * tileB[k][localCol];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (globalRow < M && globalCol < N)
    C[offsetC + globalRow * N + globalCol] = acc;
}

//===================================================================================================================//

// Tiled GEMM: C = A × B^T (no bias)
// A: (M, K), B stored as (N, K), accessed as (K, N) via transposed indexing
// C: (M, N)
kernel void gemm_dFilters(global TYPE* A, global TYPE* B, global TYPE* C, ulong offsetA, ulong offsetB, ulong offsetC,
                        ulong M, ulong N, ulong K)
{
  uint localRow = get_local_id(1);
  uint localCol = get_local_id(0);
  uint globalRow = get_group_id(1) * TILE_SIZE + localRow;
  uint globalCol = get_group_id(0) * TILE_SIZE + localCol;

  local TYPE tileA[TILE_SIZE][TILE_SIZE];
  local TYPE tileB[TILE_SIZE][TILE_SIZE];

  TYPE acc = (TYPE)0;

  uint numTilesK = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (uint t = 0; t < numTilesK; t++) {
    uint aCol = t * TILE_SIZE + localCol;
    uint bRow = t * TILE_SIZE + localRow;

    // Load A[globalRow][aCol]
    if (globalRow < M && aCol < K)
      tileA[localRow][localCol] = A[offsetA + globalRow * K + aCol];
    else
      tileA[localRow][localCol] = (TYPE)0;

    // Load B^T[bRow][globalCol] = B[globalCol][bRow]
    if (bRow < K && globalCol < N)
      tileB[localRow][localCol] = B[offsetB + globalCol * K + bRow];
    else
      tileB[localRow][localCol] = (TYPE)0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < TILE_SIZE; k++)
      acc += tileA[localRow][k] * tileB[k][localCol];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (globalRow < M && globalCol < N)
    C[offsetC + globalRow * N + globalCol] = acc;
}

//===================================================================================================================//

// Backward weight-gradient GEMM with K-parallel reduction: dFilters = dOut × im2col^T
// (equivalent to gemm_dFilters, but parallelized along K instead of the output grid).
//
// WHY: the dFilters output shape is M=numFilters (small) by N=C_in*kH*kW (small), while
// K=outH*outW (huge). The tiled gemm_dFilters maps one work-item per output element and so
// launches only ceil(M/16)*ceil(N/16) work-groups — far too few to occupy the GPU (e.g. the
// first conv layer: M=32, N=27 -> 4 work-groups). Each work-item then grinds through the
// whole K reduction serially. The output M*N is too small to ever fill the device via output
// tiling, so the only available source of parallelism is the K dimension.
//
// HOW: one work-group per OUTPUT element (oc, ic). The local work-items cooperatively stride
// over K, each accumulating a partial dot product dOut[oc,s]*im2col[ic,s], then tree-reduce in
// local memory. Consecutive work-items read consecutive s -> coalesced global-memory access.
// Each group writes its single output element exactly once (full assignment, like gemm_dFilters),
// so no zeroing of dFilters is required.
//
// Dispatched only for conv layers with large K; layers with small K and a large M*N stay on
// gemm_dFilters (the tree reduction here would dominate when K is short). See CNN_GPUKernelBuilder.
//
// A = dOut/skip at offsetA, shape (M, K) row-major:  A[oc*K + s]
// B = im2col/skip at offsetB, shape (N, K) row-major: B[ic*K + s]
// C = dFilters at offsetC, shape (M, N):              C[oc*N + ic]
// Dispatch: global (M*N * localWS), local (localWS)  [localWS must equal the partials array size]
kernel void gemm_dFilters_kpar(global TYPE* A, global TYPE* B, global TYPE* C, ulong offsetA, ulong offsetB, ulong offsetC,
                               ulong M, ulong N, ulong K)
{
  ulong e = get_group_id(0); // output element index in [0, M*N)
  ulong oc = e / N;
  ulong ic = e % N;
  uint lid = get_local_id(0);
  uint ls = get_local_size(0);

  ulong aBase = offsetA + oc * K; // A[oc, :] row base
  ulong bBase = offsetB + ic * K; // B[ic, :] row base

  TYPE partial = (TYPE)0;

  for (ulong s = lid; s < K; s += ls)
    partial += A[aBase + s] * B[bBase + s];

  local TYPE partials[256];
  partials[lid] = partial;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint off = ls >> 1; off > 0; off >>= 1) {
    if (lid < off)
      partials[lid] += partials[lid + off];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0)
    C[offsetC + e] = partials[0];
}

#endif // CNN_GEMM_CPP_CL
