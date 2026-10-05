#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_LT(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cublasLt err %d line %d\n",s,__LINE__);exit(1);} } while(0)
#define CHECK_RT(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("cuda err %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)

// Row-major C[M,N] = A[M,K] * W[K,N] mapped to colmajor:
//   A colmajor (K,M) ld=K, opA=T ; W colmajor (N,K) ld=N, opB=T ; C colmajor (N,M) ld=N
static void bench_one(int M, int N, int K, int fp16_out, cublasLtHandle_t lt, void* ws, size_t wsSize) {
  size_t aB = (size_t)M*K*2, bB = (size_t)N*K*2;
  void *dA, *dB, *dC;
  CHECK_RT(cudaMalloc(&dA, aB));
  CHECK_RT(cudaMalloc(&dB, bB));
  CHECK_RT(cudaMalloc(&dC, (size_t)M*N*(fp16_out?2:4)));
  CHECK_RT(cudaMemset(dA, 0x3c, aB));
  CHECK_RT(cudaMemset(dB, 0x3c, bB));

  cublasLtMatmulDesc_t op;
  CHECK_LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t nN = CUBLAS_OP_N;
  CHECK_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &nN, sizeof(nN)));
  CHECK_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &nN, sizeof(nN)));

  // rowmajor C[M,N] = A[M,K] * W[K,N]  <=>  colmain C'(N,M) = W'(N,K) * A'(K,M), all op=N
  cublasLtMatrixLayout_t la, lb, lc;
  CHECK_LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, N, K, N));
  CHECK_LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, K, M, K));
  CHECK_LT(cublasLtMatrixLayoutCreate(&lc, fp16_out?CUDA_R_16F:CUDA_R_32F, N, M, N));

  cublasLtMatmulPreference_t pref;
  CHECK_LT(cublasLtMatmulPreferenceCreate(&pref));
  CHECK_LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)));
  cublasLtMatmulHeuristicResult_t hr[8]; int nr=0;
  CHECK_LT(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 8, hr, &nr));
  if (nr==0) { printf("M %d N %d K %d out %s : no algo\n", M,N,K, fp16_out?"f16":"f32"); return; }

  float one = 1.0f, zero = 0.0f;
  cublasLtMatmul(lt, op, &one, dB, la, dA, lb, &zero, dC, lc, dC, lc, &hr[0].algo, ws, wsSize, 0);
  CHECK_RT(cudaGetLastError());

  cudaEvent_t e0, e1; CHECK_RT(cudaEventCreate(&e0)); CHECK_RT(cudaEventCreate(&e1));
  for (int i=0;i<5;i++)
    cublasLtMatmul(lt, op, &one, dB, la, dA, lb, &zero, dC, lc, dC, lc, &hr[0].algo, ws, wsSize, 0);
  CHECK_RT(cudaDeviceSynchronize());
  CHECK_RT(cudaEventRecord(e0, 0));
  const int iters = 30;
  for (int i=0;i<iters;i++)
    cublasLtMatmul(lt, op, &one, dB, la, dA, lb, &zero, dC, lc, dC, lc, &hr[0].algo, ws, wsSize, 0);
  CHECK_RT(cudaEventRecord(e1, 0));
  CHECK_RT(cudaEventSynchronize(e1));
  float ms; CHECK_RT(cudaEventElapsedTime(&ms, e0, e1));
  double tflops = 2.0*M*N*K*iters/(ms*1e-3)*1e-12;
  printf("RESULT M %d N %d K %d out %s : %.4f ms  %.1f TFLOPS\n", M,N,K, fp16_out?"f16":"f32", ms/iters, tflops);
  fflush(stdout);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatmulDescDestroy(op);
}

int main(void) {
  cublasLtHandle_t lt; CHECK_LT(cublasLtCreate(&lt));
  void* ws; size_t wsSize = 64ull<<20;
  CHECK_RT(cudaMalloc(&ws, wsSize));
  int Ms[] = {16058, 71362};
  int NK[][2] = {{256,256},{768,256},{1024,256},{256,1024}};
  for (int mi=0; mi<2; mi++)
    for (int ni=0; ni<4; ni++) {
      bench_one(Ms[mi], NK[ni][0], NK[ni][1], 1, lt, ws, wsSize);
      bench_one(Ms[mi], NK[ni][0], NK[ni][1], 0, lt, ws, wsSize);
    }
  printf("ALL-DONE\n");
  return 0;
}
