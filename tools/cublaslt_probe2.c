#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CK(x) do { cublasStatus_t s=(x); if(s!=0){printf("LT err %d line %d\n",s,__LINE__); return 1;} } while(0)
#define RT(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("cuda err %s line %d\n",cudaGetErrorString(e),__LINE__); return 1;} } while(0)

static int bench_one(cublasLtHandle_t lt, int M, int N, int K, int f16_out, void* ws, size_t wsSize, int nalgos) {
  void *dW, *dX, *dY, *dC, *dBias;
  RT(cudaMalloc(&dW, (size_t)N*K*2)); RT(cudaMalloc(&dX, (size_t)M*K*2));
  RT(cudaMalloc(&dY, (size_t)M*N*(f16_out?2:4))); RT(cudaMalloc(&dC, (size_t)M*N*4));
  RT(cudaMalloc(&dBias, (size_t)N*(f16_out?2:4)));
  RT(cudaMemset(dW,0x3c,(size_t)N*K*2)); RT(cudaMemset(dX,0x3c,(size_t)M*K*2));
  printf("M%d N%d K%d out%s bias on beta=1:\n", M,N,K, f16_out?"f16":"f32");
  cublasLtMatmulDesc_t op; CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t tN=CUBLAS_OP_N, tT=CUBLAS_OP_T;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tT, sizeof(tT)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tN, sizeof(tN)));
  cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dBias, sizeof(dBias)));
  cublasLtMatrixLayout_t la,lb,lc,ld;
  CK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, K, N, K));
  CK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, K, M, K));
  cudaDataType_t dt = f16_out?CUDA_R_16F:CUDA_R_32F;
  CK(cublasLtMatrixLayoutCreate(&lc, dt, N, M, N));
  CK(cublasLtMatrixLayoutCreate(&ld, dt, N, M, N));
  cublasLtMatmulPreference_t pref; CK(cublasLtMatmulPreferenceCreate(&pref));
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)));
  cublasLtMatmulHeuristicResult_t hr[16]; int nr=0;
  CK(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, ld, pref, nalgos, hr, &nr));
  printf("  got %d algos\n", nr);
  float one=1.0f, beta=1.0f;
  cudaEvent_t e0,e1; RT(cudaEventCreate(&e0)); RT(cudaEventCreate(&e1));
  for (int i=0;i<nr;i++) {
    if (hr[i].state != 0) { printf("  algo %d: state %d\n", i, hr[i].state); continue; }
    for (int w=0;w<3;w++) cublasLtMatmul(lt, op, &one, dW, la, dX, lb, &beta, dC, lc, dY, ld, &hr[i].algo, ws, wsSize, 0);
    RT(cudaDeviceSynchronize());
    RT(cudaEventRecord(e0, 0));
    for (int w=0;w<30;w++) cublasLtMatmul(lt, op, &one, dW, la, dX, lb, &beta, dC, lc, dY, ld, &hr[i].algo, ws, wsSize, 0);
    RT(cudaEventRecord(e1, 0));
    RT(cudaEventSynchronize(e1));
    float ms; RT(cudaEventElapsedTime(&ms, e0, e1));
    double tf = 2.0*M*N*K*30/(ms*1e-3)*1e-12;
    printf("  algo %d: %.1f us  %.1f TFLOPS  ws %zu  waves %.2f\n", i, ms*1000.0/30, tf, hr[i].workspaceSize, hr[i].wavesCount);
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
  // resid shapes (both chunk sizes)
  bench_one(lt, 16058, 256, 512, 0, ws, wsSize, 8);
  bench_one(lt, 71362, 256, 512, 0, ws, wsSize, 8);
  bench_one(lt, 16058, 256, 1024, 0, ws, wsSize, 8);
  bench_one(lt, 71362, 256, 1024, 0, ws, wsSize, 8);
  // qkv shapes
  bench_one(lt, 16058, 1536, 256, 1, ws, wsSize, 8);
  bench_one(lt, 71362, 1536, 256, 1, ws, wsSize, 8);
  printf("PROBE2-DONE\n");
  return 0;
}
