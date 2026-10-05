#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CK(x) do { cublasStatus_t s=(x); if(s!=0){printf("  -> LT err %d at line %d\n",s,__LINE__); return 1;} } while(0)
#define RT(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("  -> cuda err %s line %d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)

// Replicates our Rust call: rowmajor y[M,N] = x16[M,K] f16 * W[N,K]^T + bias (+ beta*C)
// colmain: D(N,M) ld=N = opT(Wc(K,N) ld=K) * opN(Xc(K,M) ld=K)
static int probe(cublasLtHandle_t lt, int M, int N, int K, int f16_out, int bias_f32, int use_bias, float beta, void* ws, size_t wsSize) {
  printf("M%d N%d K%d out%s bias=%s(%s) beta=%.0f :", M,N,K, f16_out?"f16":"f32", use_bias?"on":"off", bias_f32?"f32":"f16", beta);
  void *dW, *dX, *dY, *dC, *dBias;
  RT(cudaMalloc(&dW, (size_t)N*K*2)); RT(cudaMalloc(&dX, (size_t)M*K*2));
  RT(cudaMalloc(&dY, (size_t)M*N*(f16_out?2:4))); RT(cudaMalloc(&dC, (size_t)M*N*4));
  RT(cudaMalloc(&dBias, (size_t)N*(bias_f32?4:2)));
  RT(cudaMemset(dW,0x3c,(size_t)N*K*2)); RT(cudaMemset(dX,0x3c,(size_t)M*K*2));

  cublasLtMatmulDesc_t op; CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t tN=CUBLAS_OP_N, tT=CUBLAS_OP_T;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tT, sizeof(tT)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tN, sizeof(tN)));
  if (use_bias) {
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    if (bias_f32) {
      cudaDataType_t bt = CUDA_R_32F;
      CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)));
    }
    CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dBias, sizeof(dBias)));
  }
  cublasLtMatrixLayout_t la,lb,lc,ld;
  CK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, K, N, K));
  CK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, K, M, K));
  cudaDataType_t dt = f16_out?CUDA_R_16F:CUDA_R_32F;
  CK(cublasLtMatrixLayoutCreate(&lc, dt, N, M, N));
  CK(cublasLtMatrixLayoutCreate(&ld, dt, N, M, N));
  cublasLtMatmulPreference_t pref; CK(cublasLtMatmulPreferenceCreate(&pref));
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)));
  cublasLtMatmulHeuristicResult_t hr[4]; int nr=0;
  cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, ld, pref, 4, hr, &nr);
  if (hs!=0) { printf("heuristic status %d\n", hs); }
  else if (nr==0) printf("no algo\n");
  else {
    float one=1.0f, bt=beta;
    cublasStatus_t ms = cublasLtMatmul(lt, op, &one, dW, la, dX, lb, &bt, dC, lc, dY, ld, &hr[0].algo, ws, wsSize, 0);
    RT(cudaGetLastError());
    RT(cudaDeviceSynchronize());
    printf("OK %d algos, matmul status %d, ws %zu\n", nr, ms, hr[0].workspaceSize);
  }
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(lc); cublasLtMatrixLayoutDestroy(ld);
  cublasLtMatmulDescDestroy(op);
  cudaFree(dW); cudaFree(dX); cudaFree(dY); cudaFree(dC); cudaFree(dBias);
  return 0;
}

int main(void) {
  cublasLtHandle_t lt; CK(cublasLtCreate(&lt));
  void* ws; size_t wsSize = 32ull<<20; RT(cudaMalloc(&ws, wsSize));
  int M=16058;
  // step 1: plain NT, f32 out, beta 0
  probe(lt, M, 256, 512, 0, 1, 0, 0, ws, wsSize);
  // step 2: + BIAS epilogue, f32 out, f32 bias (the resid path)
  probe(lt, M, 256, 512, 0, 1, 1, 0, ws, wsSize);
  // step 3: + beta=1
  probe(lt, M, 256, 512, 0, 1, 1, 1, ws, wsSize);
  // step 4: f16 out + BIAS, bias dtype f16 (default)
  probe(lt, M, 1536, 256, 1, 0, 1, 0, ws, wsSize);
  // step 5: f16 out + BIAS + BIAS_DATA_TYPE f32 (the qkv path)
  probe(lt, M, 1536, 256, 1, 1, 1, 0, ws, wsSize);
  printf("PROBE-DONE\n");
  return 0;
}
