// Microbenchmark kernels for DGX Spark (GB10, sm_121).
// Built at runtime by scripts/bench_spark.py via torch.utils.cpp_extension.load_inline.
//
//   time_read           streaming read bandwidth (DRAM, or L2 when the buffer fits)
//   time_coop_barrier   cooperative-groups grid.sync() latency
//   time_flag_barrier   atomic-counter grid barrier latency (what megakernels use)
//   nvfp4_gemv          y[M,N] = x[M,K] @ dequant(W)[N,K]^T with NVFP4 weights:
//                       packed E2M1 nibbles + one E4M3 scale per 16 values + fp32 global scale

#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace cg = cooperative_groups;

#define CUDA_OK(x) TORCH_CHECK((x) == cudaSuccess, "CUDA error: ", cudaGetErrorString(cudaGetLastError()))

struct EventTimer {
  cudaEvent_t a, b;
  cudaStream_t s;
  explicit EventTimer(cudaStream_t s_) : s(s_) {
    cudaEventCreate(&a);
    cudaEventCreate(&b);
  }
  void start() { cudaEventRecord(a, s); }
  double stop_ms() {
    cudaEventRecord(b, s);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    return ms;
  }
  ~EventTimer() {
    cudaEventDestroy(a);
    cudaEventDestroy(b);
  }
};

// ------------------------------------------------------------ streaming read
__global__ void read_kernel(const int4* __restrict__ p, long long n, int iters, unsigned* sink) {
  unsigned acc = 0;
  const long long stride = (long long)gridDim.x * blockDim.x;
  for (int it = 0; it < iters; ++it) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
      int4 v = __ldcg(p + i);  // cache in L2 only, bypass L1
      acc ^= (unsigned)(v.x ^ v.y ^ v.z ^ v.w);
    }
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;  // keeps the loads alive
}

double time_read(torch::Tensor buf, int64_t iters, int64_t blocks, int64_t threads) {
  TORCH_CHECK(buf.is_cuda() && buf.is_contiguous());
  auto stream = at::cuda::getCurrentCUDAStream();
  long long n = (long long)(buf.numel() * buf.element_size() / sizeof(int4));
  auto sink = torch::zeros({1}, buf.options().dtype(torch::kInt32));
  auto* p = reinterpret_cast<const int4*>(buf.data_ptr());
  auto* s = reinterpret_cast<unsigned*>(sink.data_ptr());
  read_kernel<<<blocks, threads, 0, stream>>>(p, n, 1, s);  // warm-up (also fills L2)
  CUDA_OK(cudaGetLastError());
  EventTimer t(stream);
  t.start();
  read_kernel<<<blocks, threads, 0, stream>>>(p, n, (int)iters, s);
  double ms = t.stop_ms();
  CUDA_OK(cudaGetLastError());
  return ms;
}

// ------------------------------------------------------------ grid barriers
__global__ void coop_barrier_kernel(int n, unsigned* sink) {
  cg::grid_group g = cg::this_grid();
  unsigned x = threadIdx.x;
  for (int i = 0; i < n; ++i) {
    x = x * 1664525u + 1013904223u;
    g.sync();
  }
  if (x == 0xdeadbeefu) sink[0] = x;
}

double time_coop_barrier(int64_t n, int64_t blocks, int64_t threads) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto sink = torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  unsigned* s = reinterpret_cast<unsigned*>(sink.data_ptr());
  int ni = (int)n;
  void* args[] = {&ni, &s};
  CUDA_OK(cudaLaunchCooperativeKernel((void*)coop_barrier_kernel, dim3(blocks), dim3(threads), args, 0, stream));
  EventTimer t(stream);
  t.start();
  CUDA_OK(cudaLaunchCooperativeKernel((void*)coop_barrier_kernel, dim3(blocks), dim3(threads), args, 0, stream));
  return t.stop_ms();
}

// Sense-reversing barrier on a global counter: thread 0 of each block
// arrives; the last arriver resets the counter and bumps the generation.
__global__ void flag_barrier_kernel(int n, unsigned* count, unsigned* gen, unsigned* sink) {
  const unsigned nb = gridDim.x;
  volatile unsigned* vgen = gen;
  unsigned x = threadIdx.x;
  for (int i = 0; i < n; ++i) {
    x = x * 1664525u + 1013904223u;
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned g = *vgen;
      __threadfence();
      if (atomicAdd(count, 1u) == nb - 1) {
        atomicExch(count, 0u);
        __threadfence();
        atomicAdd(gen, 1u);
      } else {
        while (*vgen == g) {
        }
      }
      __threadfence();
    }
    __syncthreads();
  }
  if (x == 0xdeadbeefu) sink[0] = x;
}

double time_flag_barrier(int64_t n, int64_t blocks, int64_t threads) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto state = torch::zeros({3}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  unsigned* base = reinterpret_cast<unsigned*>(state.data_ptr());
  unsigned *count = base, *gen = base + 1, *sink = base + 2;
  int ni = (int)n;
  void* args[] = {&ni, &count, &gen, &sink};
  // cooperative launch guarantees every block is co-resident (required for spinning)
  CUDA_OK(cudaLaunchCooperativeKernel((void*)flag_barrier_kernel, dim3(blocks), dim3(threads), args, 0, stream));
  EventTimer t(stream);
  t.start();
  CUDA_OK(cudaLaunchCooperativeKernel((void*)flag_barrier_kernel, dim3(blocks), dim3(threads), args, 0, stream));
  return t.stop_ms();
}

// ------------------------------------------------------------- NVFP4 GEMV
// W: uint8 [N, K/2]   two E2M1 values per byte, low nibble = even k
// S: uint8 [N, K/16]  E4M3 block scales
// X: half  [M, K]     activations (M = tokens verified per step)
// Y: float [M, N]
// One warp per output row; each lane streams 16-byte chunks (32 values,
// 2 scale blocks) and dequantizes in registers. This is the v0 decode
// GEMV; M > ~4 will want the tensor-core (mma) path in the real engine.
__device__ __forceinline__ float e4m3_to_float(uint8_t v) {
  __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)v, __NV_E4M3);
  return __half2float(__half(h));
}

template <int M>
__global__ void nvfp4_gemv_kernel(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                  const __half* __restrict__ X, float* __restrict__ Y, int N, int K,
                                  float gscale) {
  __shared__ float lut[16];
  if (threadIdx.x < 16) {
    const int i = threadIdx.x, m = i & 7;
    const float mag = m < 4 ? 0.5f * m : (m == 4 ? 2.f : (m == 5 ? 3.f : (m == 6 ? 4.f : 6.f)));
    lut[i] = (i & 8) ? -mag : mag;
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * (blockDim.x >> 5) + warp;
  if (n >= N) return;
  const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)n * (K / 2));
  const uint8_t* srow = S + (size_t)n * (K / 16);
  const int chunks = K / 32;
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  for (int c = lane; c < chunks; c += 32) {
    const uint4 w = __ldcs(wrow + c);  // streamed once: evict-first
    const float s0 = e4m3_to_float(srow[2 * c]);
    const float s1 = e4m3_to_float(srow[2 * c + 1]);
    const uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float s = q < 2 ? s0 : s1;
      const uint32_t u = words[q];
      const int kb = c * 32 + q * 8;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float wv = lut[(u >> (4 * j)) & 0xF] * s;
#pragma unroll
        for (int m = 0; m < M; ++m) acc[m] = fmaf(wv, __half2float(X[m * K + kb + j]), acc[m]);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float v = acc[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0) Y[(size_t)m * N + n] = v * gscale;
  }
}

torch::Tensor nvfp4_gemv(torch::Tensor W, torch::Tensor S, torch::Tensor X, double gscale) {
  TORCH_CHECK(W.is_cuda() && S.is_cuda() && X.is_cuda());
  TORCH_CHECK(W.scalar_type() == torch::kUInt8 && S.scalar_type() == torch::kUInt8);
  TORCH_CHECK(X.scalar_type() == torch::kHalf && X.is_contiguous() && W.is_contiguous() && S.is_contiguous());
  const int N = (int)W.size(0), K = (int)W.size(1) * 2, M = (int)X.size(0);
  TORCH_CHECK(X.size(1) == K && K % 32 == 0 && S.size(0) == N && S.size(1) == K / 16);
  auto Y = torch::empty({M, N}, X.options().dtype(torch::kFloat32));
  const int warps = 8;
  dim3 block(warps * 32), grid((N + warps - 1) / warps);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto w = W.data_ptr<uint8_t>();
  auto s = S.data_ptr<uint8_t>();
  auto x = reinterpret_cast<const __half*>(X.data_ptr<at::Half>());
  auto y = Y.data_ptr<float>();
  const float g = (float)gscale;
  switch (M) {
    case 1: nvfp4_gemv_kernel<1><<<grid, block, 0, stream>>>(w, s, x, y, N, K, g); break;
    case 2: nvfp4_gemv_kernel<2><<<grid, block, 0, stream>>>(w, s, x, y, N, K, g); break;
    case 4: nvfp4_gemv_kernel<4><<<grid, block, 0, stream>>>(w, s, x, y, N, K, g); break;
    case 8: nvfp4_gemv_kernel<8><<<grid, block, 0, stream>>>(w, s, x, y, N, K, g); break;
    case 13: nvfp4_gemv_kernel<13><<<grid, block, 0, stream>>>(w, s, x, y, N, K, g); break;
    default: TORCH_CHECK(false, "nvfp4_gemv: M must be one of 1, 2, 4, 8, 13");
  }
  CUDA_OK(cudaGetLastError());
  return Y;
}
