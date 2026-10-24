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

#include <algorithm>

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

// ======================================================================
// v1: tensor-core NVFP4 GEMV + persistent whole-step weight stream
// ======================================================================
//
// Shuffled ("fragment-major") NVFP4 layout, produced by pack_nvfp4_mma() in
// scripts/bench_spark.py. A matrix [N, K] (N padded to 16, K % 128 == 0) is
// cut into tiles of 16 rows x 64 k. Each tile is 512 weight bytes: lane L of
// a warp owns 16 contiguous bytes = 4 words, word j holding the 8 nibbles of
// its mma.m16n8k16 A-fragment for k16 block j, in register order
//   (r, k) (r, k+1) (r+8, k) (r+8, k+1) (r, k+8) (r, k+9) (r+8, k+8) (r+8, k+9)
// with r = tile_row0 + L/4 and k = block_k0 + 2*(L%4). Tiles are stored
// row-tile major. Scales: 64 bytes per tile laid out [16 rows][4 k16 blocks].

#define KCH 2  // 64-k tiles per warp task -> 128 k per task (split-K)

__device__ __forceinline__ uint32_t fp4x2_to_half2_bits(uint32_t b) {
  // Two E2M1 nibbles -> two fp16 equal to value * 2^-14 (exact, incl. the
  // E2M1 subnormal 0.5 which lands on an fp16 subnormal).
  return ((b & 0x7u) << 9) | ((b & 0x8u) << 12) | ((b & 0x70u) << 21) | ((b & 0x80u) << 24);
}

__device__ __forceinline__ uint32_t dq_pair(uint32_t byte, __half2 scale2) {
  uint32_t bits = fp4x2_to_half2_bits(byte);
  __half2 h = *reinterpret_cast<__half2*>(&bits);
  h = __hmul2(h, __float2half2_rn(16384.f));  // exact: back to {0, .5, ..., 6}
  h = __hmul2(h, scale2);                     // exact: <= 6 significant bits, <= 2688
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ __half2 e4m3_to_half2(uint32_t byte) {
  __half_raw r = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)(byte & 0xFFu), __NV_E4M3);
  return __half2half2(__half(r));
}

__device__ __forceinline__ void mma_16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// One warp task: 16 output rows x 128 k, all tokens. Xs is the activation
// tile in shared memory, [NTOK*8 rows][Kp] fp16, zero beyond the real tokens.
template <int NTOK>
__device__ __forceinline__ void mma_task(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                         const __half* Xs, int Kp, float* __restrict__ Y, int N, int K, int M,
                                         int nt, int kc, float gscale, int lane) {
  const int KT = K / 64;
  const int kt0 = kc * KCH;
  const int g = lane >> 2, c = lane & 3;
  uint4 w[KCH];
  uint32_t sg[KCH], sg8[KCH];
#pragma unroll
  for (int i = 0; i < KCH; ++i) {  // issue every load before any math
    const size_t tile = (size_t)nt * KT + kt0 + i;
    w[i] = __ldcs(reinterpret_cast<const uint4*>(W + tile * 512) + lane);
    const unsigned int* sp = reinterpret_cast<const unsigned int*>(S + tile * 64);
    sg[i] = __ldcs(sp + g);
    sg8[i] = __ldcs(sp + g + 8);
  }
  float acc[NTOK][4];
#pragma unroll
  for (int t = 0; t < NTOK; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
#pragma unroll
  for (int i = 0; i < KCH; ++i) {
    const uint32_t words[4] = {w[i].x, w[i].y, w[i].z, w[i].w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const __half2 s0 = e4m3_to_half2(sg[i] >> (8 * j));
      const __half2 s1 = e4m3_to_half2(sg8[i] >> (8 * j));
      const uint32_t u = words[j];
      uint32_t a[4];
      a[0] = dq_pair(u & 0xFFu, s0);
      a[1] = dq_pair((u >> 8) & 0xFFu, s1);
      a[2] = dq_pair((u >> 16) & 0xFFu, s0);
      a[3] = dq_pair(u >> 24, s1);
      const int k0 = (kt0 + i) * 64 + j * 16 + 2 * c;
#pragma unroll
      for (int t = 0; t < NTOK; ++t) {
        const __half* xr = Xs + (t * 8 + g) * Kp + k0;
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(xr);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(xr + 8);
        mma_16816(acc[t], a, b0, b1);
      }
    }
  }
  const int r0 = nt * 16 + g, r1 = r0 + 8;
#pragma unroll
  for (int t = 0; t < NTOK; ++t) {
    const int t0 = t * 8 + 2 * c, t1 = t0 + 1;
    if (r0 < N) {
      if (t0 < M) atomicAdd(Y + (size_t)t0 * N + r0, acc[t][0] * gscale);
      if (t1 < M) atomicAdd(Y + (size_t)t1 * N + r0, acc[t][1] * gscale);
    }
    if (r1 < N) {
      if (t0 < M) atomicAdd(Y + (size_t)t0 * N + r1, acc[t][2] * gscale);
      if (t1 < M) atomicAdd(Y + (size_t)t1 * N + r1, acc[t][3] * gscale);
    }
  }
}

template <int NTOK>
__device__ __forceinline__ void load_x(__half* xs, const __half* __restrict__ X, int M, int K, int Kp) {
  const int rows = NTOK * 8;
  for (int idx = threadIdx.x; idx < rows * (K / 2); idx += blockDim.x) {
    const int r = idx / (K / 2), k2 = idx % (K / 2);
    const __half2 v = r < M ? reinterpret_cast<const __half2*>(X + (size_t)r * K)[k2] : __float2half2_rn(0.f);
    reinterpret_cast<__half2*>(xs + r * Kp)[k2] = v;
  }
}

template <int NTOK>
__global__ void __launch_bounds__(256) mma_gemv_kernel(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                                       const __half* __restrict__ X, float* __restrict__ Y, int N,
                                                       int K, int M, float gscale) {
  extern __shared__ __align__(16) __half xs[];
  const int Kp = K + 8;  // pad rows by 16 bytes: conflict-free B-fragment loads
  load_x<NTOK>(xs, X, M, K, Kp);
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarps = (gridDim.x * blockDim.x) >> 5;
  const int kchunks = K / 64 / KCH;
  const int total = ((N + 15) / 16) * kchunks;
  for (int task = warp; task < total; task += nwarps)
    mma_task<NTOK>(W, S, xs, Kp, Y, N, K, M, task / kchunks, task % kchunks, gscale, lane);
}

// Y must be zero on entry (split-K partial sums are accumulated atomically).
torch::Tensor nvfp4_mma_gemv(torch::Tensor W, torch::Tensor S, torch::Tensor X, int64_t N, double gscale,
                             torch::Tensor Y) {
  TORCH_CHECK(W.is_cuda() && S.is_cuda() && X.is_cuda() && Y.is_cuda());
  TORCH_CHECK(X.scalar_type() == torch::kHalf && Y.scalar_type() == torch::kFloat32);
  const int M = (int)X.size(0), K = (int)X.size(1);
  TORCH_CHECK(K % (64 * KCH) == 0 && M >= 1 && M <= 16, "need K % 128 == 0 and 1 <= M <= 16");
  const int Np = ((int)N + 15) / 16 * 16;
  TORCH_CHECK(W.numel() == (int64_t)Np * K / 2 && S.numel() == (int64_t)Np * K / 16, "bad packed sizes");
  const int ntok = M <= 8 ? 1 : 2;
  const int smem = ntok * 8 * (K + 8) * (int)sizeof(__half);
  const int tasks = (Np / 16) * (K / 64 / KCH);
  const int blocks = std::max(1, std::min((tasks + 7) / 8, 48 * 8));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto x = reinterpret_cast<const __half*>(X.data_ptr<at::Half>());
  if (ntok == 1) {
    cudaFuncSetAttribute(mma_gemv_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    mma_gemv_kernel<1><<<blocks, 256, smem, stream>>>(W.data_ptr<uint8_t>(), S.data_ptr<uint8_t>(), x,
                                                      Y.data_ptr<float>(), (int)N, K, M, (float)gscale);
  } else {
    cudaFuncSetAttribute(mma_gemv_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    mma_gemv_kernel<2><<<blocks, 256, smem, stream>>>(W.data_ptr<uint8_t>(), S.data_ptr<uint8_t>(), x,
                                                      Y.data_ptr<float>(), (int)N, K, M, (float)gscale);
  }
  CUDA_OK(cudaGetLastError());
  return Y;
}

// ------------------------------------------------ persistent whole step
// One cooperative launch runs every stage of a decode step. A stage is a
// group of matrices that read the same input (e.g. GDN qkv + gate + ab);
// stages are separated by grid.sync(), exactly like the real megakernel.
struct MatDesc {
  long long W, S, X, Y;
  int N, K;
  int stage, task_off;  // task_off: first task id of this matrix within its stage
};

template <int NTOK>
__global__ void __launch_bounds__(512, 1) stream_kernel(const MatDesc* __restrict__ mats,
                                                        const int* __restrict__ stage_first, int n_stages, int M,
                                                        int iters) {
  extern __shared__ __align__(16) __half xs[];
  cg::grid_group grid = cg::this_grid();
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarps = (gridDim.x * blockDim.x) >> 5;
  for (int it = 0; it < iters; ++it) {
    for (int s = 0; s < n_stages; ++s) {
      const int m0 = stage_first[s], m1 = stage_first[s + 1];
      const int K = mats[m0].K, Kp = K + 8;
      load_x<NTOK>(xs, reinterpret_cast<const __half*>(mats[m0].X), M, K, Kp);
      __syncthreads();
      const MatDesc& last = mats[m1 - 1];
      const int total = last.task_off + ((last.N + 15) / 16) * (K / 64 / KCH);
      for (int task = warp; task < total; task += nwarps) {
        int mi = m0;
        while (mi + 1 < m1 && mats[mi + 1].task_off <= task) ++mi;
        const MatDesc& d = mats[mi];
        const int local = task - d.task_off, kchunks = K / 64 / KCH;
        mma_task<NTOK>(reinterpret_cast<const uint8_t*>(d.W), reinterpret_cast<const uint8_t*>(d.S), xs, Kp,
                       reinterpret_cast<float*>(d.Y), d.N, K, M, local / kchunks, local % kchunks, 1.f, lane);
      }
      grid.sync();
    }
  }
}

static double launch_stream(torch::Tensor descs, torch::Tensor stage_first, int64_t n_stages, int64_t M,
                            int64_t iters, int64_t kmax, bool timed) {
  TORCH_CHECK(descs.is_cuda() && descs.scalar_type() == torch::kInt64 && descs.size(1) == 6);
  TORCH_CHECK(stage_first.is_cuda() && stage_first.scalar_type() == torch::kInt32);
  TORCH_CHECK(M >= 1 && M <= 16);
  auto stream = at::cuda::getCurrentCUDAStream();
  int dev = 0;
  cudaGetDevice(&dev);
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const int ntok = M <= 8 ? 1 : 2;
  const int smem = ntok * 8 * ((int)kmax + 8) * (int)sizeof(__half);
  const MatDesc* mats = reinterpret_cast<const MatDesc*>(descs.data_ptr<int64_t>());
  const int* sf = stage_first.data_ptr<int>();
  int ns = (int)n_stages, m = (int)M, its = (int)iters;
  void* args[] = {&mats, &sf, &ns, &m, &its};
  void* fn = ntok == 1 ? (void*)stream_kernel<1> : (void*)stream_kernel<2>;
  CUDA_OK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  if (!timed) {
    CUDA_OK(cudaLaunchCooperativeKernel(fn, dim3(sms), dim3(512), args, smem, stream));
    return 0.0;
  }
  EventTimer t(stream);
  t.start();
  CUDA_OK(cudaLaunchCooperativeKernel(fn, dim3(sms), dim3(512), args, smem, stream));
  return t.stop_ms();
}

void run_stream(torch::Tensor descs, torch::Tensor stage_first, int64_t n_stages, int64_t M, int64_t kmax) {
  launch_stream(descs, stage_first, n_stages, M, 1, kmax, false);
}

double time_stream(torch::Tensor descs, torch::Tensor stage_first, int64_t n_stages, int64_t M, int64_t iters,
                   int64_t kmax) {
  launch_stream(descs, stage_first, n_stages, M, 1, kmax, false);  // warm-up
  return launch_stream(descs, stage_first, n_stages, M, iters, kmax, true);
}

// ======================================================================
// v2: software-pipelined persistent stream
// ======================================================================
// Run 2 showed v1 loses ~3 us per stage: every stage waits for DRAM latency,
// the barrier and the activation load before its first weight bytes arrive.
// Weights do not depend on activations, so v2 keeps exactly one chunk of
// loads in flight per warp at all times, across task AND stage boundaries:
// while chunk i is being multiplied, chunk i+1 (possibly in a later stage)
// is already on its way. Split-K is sized per stage on the host so tasks
// roughly match the warp count, and POLICY selects the load cache hint:
//   0 = ld.global.cs (evict-first), 1 = ld.global.cg (L2), 2 = ld.global.nc

struct TaskDesc {
  int mat, nt, kc0, nkc;  // kc in 128-wide chunks
};

struct Chunk {
  uint4 w[KCH];
  uint32_t sg[KCH], sg8[KCH];
};

template <int POLICY>
__device__ __forceinline__ uint4 ld_w(const uint4* p) {
  if constexpr (POLICY == 0) return __ldcs(p);
  else if constexpr (POLICY == 1) return __ldcg(p);
  else return __ldg(p);
}

template <int POLICY>
__device__ __forceinline__ uint32_t ld_u(const unsigned int* p) {
  if constexpr (POLICY == 0) return __ldcs(p);
  else if constexpr (POLICY == 1) return __ldcg(p);
  else return __ldg(p);
}

template <int POLICY>
__device__ __forceinline__ void load_chunk(Chunk& ch, const MatDesc& d, int nt, int kc, int lane) {
  const int KT = d.K / 64, g = lane >> 2;
  const uint8_t* W = reinterpret_cast<const uint8_t*>(d.W);
  const uint8_t* S = reinterpret_cast<const uint8_t*>(d.S);
#pragma unroll
  for (int i = 0; i < KCH; ++i) {
    const size_t tile = (size_t)nt * KT + kc * KCH + i;
    ch.w[i] = ld_w<POLICY>(reinterpret_cast<const uint4*>(W + tile * 512) + lane);
    const unsigned int* sp = reinterpret_cast<const unsigned int*>(S + tile * 64);
    ch.sg[i] = ld_u<POLICY>(sp + g);
    ch.sg8[i] = ld_u<POLICY>(sp + g + 8);
  }
}

template <int NTOK>
__device__ __forceinline__ void compute_chunk(const Chunk& ch, float (&acc)[NTOK][4], const __half* Xs, int Kp,
                                              int kc, int lane) {
  const int g = lane >> 2, c = lane & 3;
#pragma unroll
  for (int i = 0; i < KCH; ++i) {
    const uint32_t words[4] = {ch.w[i].x, ch.w[i].y, ch.w[i].z, ch.w[i].w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const __half2 s0 = e4m3_to_half2(ch.sg[i] >> (8 * j));
      const __half2 s1 = e4m3_to_half2(ch.sg8[i] >> (8 * j));
      const uint32_t u = words[j];
      uint32_t a[4];
      a[0] = dq_pair(u & 0xFFu, s0);
      a[1] = dq_pair((u >> 8) & 0xFFu, s1);
      a[2] = dq_pair((u >> 16) & 0xFFu, s0);
      a[3] = dq_pair(u >> 24, s1);
      const int k0 = (kc * KCH + i) * 64 + j * 16 + 2 * c;
#pragma unroll
      for (int t = 0; t < NTOK; ++t) {
        const __half* xr = Xs + (t * 8 + g) * Kp + k0;
        mma_16816(acc[t], a, *reinterpret_cast<const uint32_t*>(xr), *reinterpret_cast<const uint32_t*>(xr + 8));
      }
    }
  }
}

template <int NTOK>
__device__ __forceinline__ void flush_acc(float (&acc)[NTOK][4], const MatDesc& d, int nt, int M, int lane) {
  const int g = lane >> 2, c = lane & 3;
  float* Y = reinterpret_cast<float*>(d.Y);
  const int N = d.N, r0 = nt * 16 + g, r1 = r0 + 8;
#pragma unroll
  for (int t = 0; t < NTOK; ++t) {
    const int t0 = t * 8 + 2 * c, t1 = t0 + 1;
    if (r0 < N) {
      if (t0 < M) atomicAdd(Y + (size_t)t0 * N + r0, acc[t][0]);
      if (t1 < M) atomicAdd(Y + (size_t)t1 * N + r0, acc[t][1]);
    }
    if (r1 < N) {
      if (t0 < M) atomicAdd(Y + (size_t)t0 * N + r1, acc[t][2]);
      if (t1 < M) atomicAdd(Y + (size_t)t1 * N + r1, acc[t][3]);
    }
  }
}

// Advance (s, j) to the first stage >= s in which this warp owns task j.
__device__ __forceinline__ bool seek(int& s, int& j, const int* __restrict__ stf, int n_stages, int warp, int nwarps) {
  while (s < n_stages && stf[s] + warp + j * nwarps >= stf[s + 1]) {
    ++s;
    j = 0;
  }
  return s < n_stages;
}

template <int NTOK, int POLICY>
__global__ void __launch_bounds__(512, 1) stream2_kernel(const MatDesc* __restrict__ mats,
                                                         const int* __restrict__ stage_first,
                                                         const TaskDesc* __restrict__ tasks,
                                                         const int* __restrict__ stf, int n_stages, int M,
                                                         int iters) {
  extern __shared__ __align__(16) __half xs[];
  cg::grid_group grid = cg::this_grid();
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarps = (gridDim.x * blockDim.x) >> 5;
  for (int it = 0; it < iters; ++it) {
    int s = 0, j = 0, c = 0;
    bool valid = seek(s, j, stf, n_stages, warp, nwarps);
    Chunk cur, nxt;
    if (valid) {
      const TaskDesc t0 = tasks[stf[s] + warp];
      load_chunk<POLICY>(cur, mats[t0.mat], t0.nt, t0.kc0, lane);
    }
    float acc[NTOK][4];
    for (int st = 0; st < n_stages; ++st) {
      const int K = mats[stage_first[st]].K, Kp = K + 8;
      load_x<NTOK>(xs, reinterpret_cast<const __half*>(mats[stage_first[st]].X), M, K, Kp);
      __syncthreads();
      while (valid && s == st) {
        const TaskDesc td = tasks[stf[s] + warp + j * nwarps];
        if (c == 0) {
#pragma unroll
          for (int t = 0; t < NTOK; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
        }
        // position of this warp's next chunk, then put its loads in flight
        int s2 = s, j2 = j, c2 = c + 1;
        if (c2 >= td.nkc) {
          c2 = 0;
          ++j2;
        }
        const bool valid2 = seek(s2, j2, stf, n_stages, warp, nwarps);
        if (valid2) {
          const TaskDesc tn = tasks[stf[s2] + warp + j2 * nwarps];
          load_chunk<POLICY>(nxt, mats[tn.mat], tn.nt, tn.kc0 + c2, lane);
        }
        compute_chunk<NTOK>(cur, acc, xs, Kp, td.kc0 + c, lane);
        if (c + 1 >= td.nkc) flush_acc<NTOK>(acc, mats[td.mat], td.nt, M, lane);
        cur = nxt;
        s = s2;
        j = j2;
        c = c2;
        valid = valid2;
      }
      grid.sync();
    }
  }
}

template <int NTOK, int POLICY>
static void* stream2_fn() {
  return (void*)stream2_kernel<NTOK, POLICY>;
}

static double launch_stream2(torch::Tensor descs, torch::Tensor stage_first, torch::Tensor tasks, torch::Tensor stf,
                             int64_t n_stages, int64_t M, int64_t iters, int64_t kmax, int64_t policy, bool timed) {
  TORCH_CHECK(descs.is_cuda() && descs.scalar_type() == torch::kInt64 && descs.size(1) == 6);
  TORCH_CHECK(tasks.is_cuda() && tasks.scalar_type() == torch::kInt32 && tasks.size(1) == 4);
  TORCH_CHECK(stage_first.scalar_type() == torch::kInt32 && stf.scalar_type() == torch::kInt32);
  TORCH_CHECK(M >= 1 && M <= 16 && policy >= 0 && policy <= 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const int ntok = M <= 8 ? 1 : 2;
  const int smem = ntok * 8 * ((int)kmax + 8) * (int)sizeof(__half);
  void* fn = nullptr;
  if (ntok == 1) fn = policy == 0 ? stream2_fn<1, 0>() : policy == 1 ? stream2_fn<1, 1>() : stream2_fn<1, 2>();
  else fn = policy == 0 ? stream2_fn<2, 0>() : policy == 1 ? stream2_fn<2, 1>() : stream2_fn<2, 2>();
  const MatDesc* mats = reinterpret_cast<const MatDesc*>(descs.data_ptr<int64_t>());
  const int* sf = stage_first.data_ptr<int>();
  const TaskDesc* tk = reinterpret_cast<const TaskDesc*>(tasks.data_ptr<int>());
  const int* tf = stf.data_ptr<int>();
  int ns = (int)n_stages, m = (int)M, its = (int)iters;
  void* args[] = {&mats, &sf, &tk, &tf, &ns, &m, &its};
  CUDA_OK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  if (!timed) {
    CUDA_OK(cudaLaunchCooperativeKernel(fn, dim3(sms), dim3(512), args, smem, stream));
    return 0.0;
  }
  EventTimer t(stream);
  t.start();
  CUDA_OK(cudaLaunchCooperativeKernel(fn, dim3(sms), dim3(512), args, smem, stream));
  return t.stop_ms();
}

void run_stream2(torch::Tensor descs, torch::Tensor stage_first, torch::Tensor tasks, torch::Tensor stf,
                 int64_t n_stages, int64_t M, int64_t kmax, int64_t policy) {
  launch_stream2(descs, stage_first, tasks, stf, n_stages, M, 1, kmax, policy, false);
}

double time_stream2(torch::Tensor descs, torch::Tensor stage_first, torch::Tensor tasks, torch::Tensor stf,
                    int64_t n_stages, int64_t M, int64_t iters, int64_t kmax, int64_t policy) {
  launch_stream2(descs, stage_first, tasks, stf, n_stages, M, 1, kmax, policy, false);  // warm-up
  return launch_stream2(descs, stage_first, tasks, stf, n_stages, M, iters, kmax, policy, true);
}

// ------------------------------------------- cache hierarchy probes
// Same streaming read as read_kernel, but each pass starts at a different
// offset so no thread ever re-reads its own addresses: rules out any
// per-thread reuse artifact in the L2 sweep.
__global__ void read_rot_kernel(const int4* __restrict__ p, long long n, int iters, unsigned* sink) {
  unsigned acc = 0;
  const long long stride = (long long)gridDim.x * blockDim.x;
  for (int it = 0; it < iters; ++it) {
    const long long off = ((long long)it * (n / 7 + 4099)) % n;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
      long long idx = i + off;
      if (idx >= n) idx -= n;
      int4 v = __ldcg(p + idx);
      acc ^= (unsigned)(v.x ^ v.y ^ v.z ^ v.w) + (unsigned)it;
    }
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

double time_read_rot(torch::Tensor buf, int64_t iters, int64_t blocks, int64_t threads) {
  auto stream = at::cuda::getCurrentCUDAStream();
  long long n = (long long)(buf.numel() * buf.element_size() / sizeof(int4));
  auto sink = torch::zeros({1}, buf.options().dtype(torch::kInt32));
  auto* p = reinterpret_cast<const int4*>(buf.data_ptr());
  auto* s = reinterpret_cast<unsigned*>(sink.data_ptr());
  read_rot_kernel<<<blocks, threads, 0, stream>>>(p, n, 1, s);
  EventTimer t(stream);
  t.start();
  read_rot_kernel<<<blocks, threads, 0, stream>>>(p, n, (int)iters, s);
  double ms = t.stop_ms();
  CUDA_OK(cudaGetLastError());
  return ms;
}

// Dependent-load latency: one thread walks a random cycle of 128-byte lines.
__global__ void chase_kernel(const unsigned* __restrict__ next, unsigned start, int hops, unsigned* sink) {
  unsigned p = start;
  for (int h = 0; h < hops; ++h) p = __ldcg(next + p);
  sink[0] = p;
}

double time_chase(torch::Tensor next, int64_t start, int64_t hops) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto sink = torch::zeros({1}, next.options());
  const unsigned* nx = reinterpret_cast<const unsigned*>(next.data_ptr<int>());
  unsigned* s = reinterpret_cast<unsigned*>(sink.data_ptr<int>());
  chase_kernel<<<1, 1, 0, stream>>>(nx, (unsigned)start, (int)std::min<int64_t>(hops, 4096), s);  // warm
  EventTimer t(stream);
  t.start();
  chase_kernel<<<1, 1, 0, stream>>>(nx, (unsigned)start, (int)hops, s);
  double ms = t.stop_ms();
  CUDA_OK(cudaGetLastError());
  return ms;
}

// ======================================================================
// v3: latency-lean persistent stream + per-stage profiling
// ======================================================================
// Run 3 showed v2 was latency-bound, not bandwidth-bound: even a 15.6 MB
// working set that fits in L2 streamed at only 194 GB/s (~5.4 us/stage).
// Causes and fixes:
//   * per-chunk dependent global loads (task table -> matrix descriptor ->
//     weights): descriptors now live in shared memory, tasks are decoded
//     arithmetically, so the next chunk's loads issue with no global round trip
//   * load_x serialized 6-16 round trips: now one vectorized round trip
//   * dequant cost: new nibble order lets each fp16 pair be built with 4 ALU
//     ops, block scales are converted two at a time and pre-scaled by 2^7 so
//     products stay exact in fp16; B fragments are one 64-bit LDS
//
// Layout v3 (pack_nvfp4_v3 in scripts/bench_spark.py), per 16x64 tile:
//   weights 512 B: lane L, word j (k16 block j): nibble r and r+4 form A
//   register r: nib0/4 = (g, 2c/2c+1), nib1/5 = (g+8, 2c/2c+1),
//   nib2/6 = (g, 2c+8/2c+9), nib3/7 = (g+8, 2c+8/2c+9); g = L/4, c = L%4
//   scales 64 B: [g 8][j 4][2] = (scale(row g, block j), scale(row g+8, block j))
// Activations in shared memory are permuted per 16-k block so lane (g, c)
// finds k = {2c, 2c+1, 2c+8, 2c+9} contiguous: one uint2 load per mma.

#define V3_MAXMATS 192
#define V3_MAXSTAGES 160
#define V3_THREADS 512

struct MatDesc3 {
  long long W, S, X, Y;
  int N, K;
  int task_off, splits;  // first task of this matrix in its stage; K-splits per row tile
  int kchunks, pad;      // K / 128
};
struct StageDesc3 {
  int m0, m1, nkc, total;  // matrices [m0, m1), chunks per task, tasks in stage
};

template <int R>
__device__ __forceinline__ uint32_t fp4_pair(uint32_t u) {
  // fp16x2 bits of (nibble R, nibble R+4) * 2^-14, exact (see fp4x2_to_half2_bits)
  uint32_t a;
  if constexpr (R < 3) a = u << (9 - 4 * R);
  else a = u >> 3;
  const uint32_t b = u << (12 - 4 * R);
  return (a & 0x0E000E00u) | (b & 0x80008000u);
}

struct Chunk3 {
  uint4 w[KCH];
  uint2 sc[KCH];
};

template <int POLICY>
__device__ __forceinline__ uint2 ld_u2(const uint2* p) {
  if constexpr (POLICY == 0) return __ldcs(p);
  else return __ldcg(p);
}

template <int POLICY>
__device__ __forceinline__ uint4 ld_w3(const uint4* p) {
  if constexpr (POLICY == 0) return __ldcs(p);
  else return __ldcg(p);
}

template <int POLICY>
__device__ __forceinline__ void load_chunk3(Chunk3& ch, const MatDesc3& d, int nt, int kc, int lane) {
  const int KT = d.K / 64, g = lane >> 2;
  const uint8_t* W = reinterpret_cast<const uint8_t*>(d.W);
  const uint8_t* S = reinterpret_cast<const uint8_t*>(d.S);
#pragma unroll
  for (int i = 0; i < KCH; ++i) {
    const size_t tile = (size_t)nt * KT + kc * KCH + i;
    ch.w[i] = ld_w3<POLICY>(reinterpret_cast<const uint4*>(W + tile * 512) + lane);
    ch.sc[i] = ld_u2<POLICY>(reinterpret_cast<const uint2*>(S + tile * 64) + g);
  }
}

template <int NTOK, int MATH>
__device__ __forceinline__ void compute_chunk3(const Chunk3& ch, float (&acc)[NTOK][4], const __half* Xs, int Kp,
                                               int kc, int lane) {
  const int g = lane >> 2, c = lane & 3;
  if constexpr (MATH == 0) {  // touch the data only: isolates memory + sync cost
#pragma unroll
    for (int i = 0; i < KCH; ++i)
      acc[0][0] += __uint_as_float((ch.w[i].x ^ ch.w[i].y ^ ch.w[i].z ^ ch.w[i].w ^ ch.sc[i].x) & 0x007FFFFFu);
    return;
  }
  const __half2 k128 = __float2half2_rn(128.f);
#pragma unroll
  for (int i = 0; i < KCH; ++i) {
    const uint32_t words[4] = {ch.w[i].x, ch.w[i].y, ch.w[i].z, ch.w[i].w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t pair = ((j < 2 ? ch.sc[i].x : ch.sc[i].y) >> (16 * (j & 1))) & 0xFFFFu;
      __half2_raw sr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)pair, __NV_E4M3);
      const __half2 s = __hmul2(__half2(sr), k128);  // (s_g, s_g8) * 2^7, exact
      const __half2 s0 = __low2half2(s), s1 = __high2half2(s);
      const uint32_t u = words[j];
      uint32_t a[4];
      uint32_t v;
      __half2 h;
      v = fp4_pair<0>(u); h = __hmul2(*reinterpret_cast<__half2*>(&v), s0); a[0] = *reinterpret_cast<uint32_t*>(&h);
      v = fp4_pair<1>(u); h = __hmul2(*reinterpret_cast<__half2*>(&v), s1); a[1] = *reinterpret_cast<uint32_t*>(&h);
      v = fp4_pair<2>(u); h = __hmul2(*reinterpret_cast<__half2*>(&v), s0); a[2] = *reinterpret_cast<uint32_t*>(&h);
      v = fp4_pair<3>(u); h = __hmul2(*reinterpret_cast<__half2*>(&v), s1); a[3] = *reinterpret_cast<uint32_t*>(&h);
      const int blk = (kc * KCH + i) * 4 + j;
#pragma unroll
      for (int t = 0; t < NTOK; ++t) {
        const uint2 b = *reinterpret_cast<const uint2*>(Xs + (t * 8 + g) * Kp + blk * 16 + c * 4);
        mma_16816(acc[t], a, b.x, b.y);
      }
    }
  }
}

template <int NTOK>
__device__ __forceinline__ void flush3(float (&acc)[NTOK][4], const MatDesc3& d, int nt, int M, int lane, float scale) {
  const int g = lane >> 2, c = lane & 3;
  float* Y = reinterpret_cast<float*>(d.Y);
  const int N = d.N, r0 = nt * 16 + g, r1 = r0 + 8;
#pragma unroll
  for (int t = 0; t < NTOK; ++t) {
    const int t0 = t * 8 + 2 * c, t1 = t0 + 1;
    if (r0 < N) {
      if (t0 < M) atomicAdd(Y + (size_t)t0 * N + r0, acc[t][0] * scale);
      if (t1 < M) atomicAdd(Y + (size_t)t1 * N + r0, acc[t][1] * scale);
    }
    if (r1 < N) {
      if (t0 < M) atomicAdd(Y + (size_t)t0 * N + r1, acc[t][2] * scale);
      if (t1 < M) atomicAdd(Y + (size_t)t1 * N + r1, acc[t][3] * scale);
    }
  }
}

// One vectorized round trip: every thread issues all its 16-byte loads, then
// scatters them into the permuted shared-memory layout. Rows >= M are zeroed.
template <int NTOK>
__device__ __forceinline__ void load_x3(__half* xs, const __half* __restrict__ X, int M, int K, int Kp) {
  constexpr int ROWS = NTOK * 8;
  constexpr int MAXV = ROWS / 2;  // ROWS * (2048 / 8) / V3_THREADS
  const int per_row = K / 8;
  const int total = ROWS * per_row;
  uint4 v[MAXV];
#pragma unroll
  for (int u = 0; u < MAXV; ++u) {
    const int idx = threadIdx.x + u * V3_THREADS;
    v[u] = make_uint4(0u, 0u, 0u, 0u);
    if (idx < total) {
      const int r = idx / per_row, q = idx - r * per_row;
      if (r < M) v[u] = __ldcg(reinterpret_cast<const uint4*>(X + (size_t)r * K) + q);
    }
  }
#pragma unroll
  for (int u = 0; u < MAXV; ++u) {
    const int idx = threadIdx.x + u * V3_THREADS;
    if (idx < total) {
      const int r = idx / per_row, q = idx - r * per_row;
      const int k = q * 8;
      uint32_t* dst = reinterpret_cast<uint32_t*>(xs + r * Kp + (k >> 4) * 16 + ((k >> 3) & 1) * 2);
      dst[0] = v[u].x;  // halves (2c, 2c+1) of this 8-run -> position c*4 (+2 for the upper run)
      dst[2] = v[u].y;
      dst[4] = v[u].z;
      dst[6] = v[u].w;
    }
  }
}

__device__ __forceinline__ void decode3(const MatDesc3* sm, const StageDesc3& sd, int t, int& mi, int& nt, int& kc0,
                                        int& n) {
  int i = sd.m0;
  while (i + 1 < sd.m1 && sm[i + 1].task_off <= t) ++i;
  const MatDesc3& d = sm[i];
  const int local = t - d.task_off;
  nt = local / d.splits;
  kc0 = (local - nt * d.splits) * sd.nkc;
  n = min(sd.nkc, d.kchunks - kc0);
  mi = i;
}

__device__ __forceinline__ long long globaltimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

template <int NTOK, int POLICY, int MATH>
__global__ void __launch_bounds__(V3_THREADS, 1) stream3_kernel(const MatDesc3* __restrict__ gm,
                                                                const StageDesc3* __restrict__ gs, int n_mats,
                                                                int n_stages, int M, int iters, float out_scale,
                                                                long long* __restrict__ prof) {
  __shared__ MatDesc3 sm[V3_MAXMATS];
  __shared__ StageDesc3 ss[V3_MAXSTAGES];
  extern __shared__ __align__(16) __half xs[];
  for (int i = threadIdx.x; i < n_mats; i += blockDim.x) sm[i] = gm[i];
  for (int i = threadIdx.x; i < n_stages; i += blockDim.x) ss[i] = gs[i];
  __syncthreads();
  cg::grid_group grid = cg::this_grid();
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarps = (gridDim.x * blockDim.x) >> 5;
  const bool rec = prof != nullptr && threadIdx.x == 0;
  if (rec && blockIdx.x == 0) {
    prof[0] = clock64();
    prof[1] = globaltimer();
  }
  for (int it = 0; it < iters; ++it) {
    // position of this warp's current chunk: stage s, j-th task of the warp in s, chunk c
    int s = 0, j = 0, c = 0;
    while (s < n_stages && warp >= ss[s].total) ++s;
    bool valid = s < n_stages;
    int mi = 0, nt = 0, kc0 = 0, n = 0;
    Chunk3 cur, nxt;
    if (valid) {
      decode3(sm, ss[s], warp, mi, nt, kc0, n);
      load_chunk3<POLICY>(cur, sm[mi], nt, kc0, lane);
    }
    float acc[NTOK][4];
    for (int st = 0; st < n_stages; ++st) {
      const long long t0 = clock64();
      const MatDesc3& first = sm[ss[st].m0];
      const int K = first.K, Kp = K + 16;
      load_x3<NTOK>(xs, reinterpret_cast<const __half*>(first.X), M, K, Kp);
      __syncthreads();
      const long long t1 = clock64();
      while (valid && s == st) {
        if (c == 0) {
#pragma unroll
          for (int t = 0; t < NTOK; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
        }
        // next chunk: same task, next task of this warp, or a later stage
        int s2 = s, j2 = j, c2 = c + 1, mi2 = mi, nt2 = nt, kc02 = kc0, n2 = n;
        if (c2 >= n) {
          c2 = 0;
          ++j2;
          while (s2 < n_stages && warp + j2 * nwarps >= ss[s2].total) {
            ++s2;
            j2 = 0;
          }
          if (s2 < n_stages) decode3(sm, ss[s2], warp + j2 * nwarps, mi2, nt2, kc02, n2);
        }
        const bool valid2 = s2 < n_stages;
        if (valid2) load_chunk3<POLICY>(nxt, sm[mi2], nt2, kc02 + c2, lane);
        compute_chunk3<NTOK, MATH>(cur, acc, xs, Kp, kc0 + c, lane);
        if (c + 1 >= n) flush3<NTOK>(acc, sm[mi], nt, M, lane, out_scale);
        cur = nxt;
        s = s2; j = j2; c = c2; mi = mi2; nt = nt2; kc0 = kc02; n = n2;
        valid = valid2;
      }
      if (prof != nullptr && it == 0) {  // uniform per launch, so every thread takes this branch
        __syncthreads();                 // whole block done with this stage
        if (rec) {
          long long* p = prof + 4 + ((long long)st * gridDim.x + blockIdx.x) * 3;
          p[0] = t0;
          p[1] = t1;
          p[2] = clock64();
        }
      }
      grid.sync();
    }
  }
  if (rec && blockIdx.x == 0) {
    prof[2] = clock64();
    prof[3] = globaltimer();
  }
}

template <int NTOK, int POLICY, int MATH>
static void* s3fn() {
  return (void*)stream3_kernel<NTOK, POLICY, MATH>;
}

static void* pick_stream3(int ntok, int policy, int math) {
  if (ntok == 1) {
    if (policy == 0) return math ? s3fn<1, 0, 1>() : s3fn<1, 0, 0>();
    return math ? s3fn<1, 1, 1>() : s3fn<1, 1, 0>();
  }
  if (policy == 0) return math ? s3fn<2, 0, 1>() : s3fn<2, 0, 0>();
  return math ? s3fn<2, 1, 1>() : s3fn<2, 1, 0>();
}

static double launch_stream3(torch::Tensor descs, torch::Tensor stages, int64_t M, int64_t iters, int64_t kmax,
                             int64_t policy, int64_t math, double out_scale, torch::Tensor prof, bool timed) {
  TORCH_CHECK(descs.is_cuda() && descs.scalar_type() == torch::kInt64 && descs.size(1) == 7);
  TORCH_CHECK(stages.is_cuda() && stages.scalar_type() == torch::kInt32 && stages.size(1) == 4);
  TORCH_CHECK(descs.size(0) <= V3_MAXMATS && stages.size(0) <= V3_MAXSTAGES);
  TORCH_CHECK(M >= 1 && M <= 16 && (policy == 0 || policy == 1));
  auto stream = at::cuda::getCurrentCUDAStream();
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const int ntok = M <= 8 ? 1 : 2;
  const int smem = ntok * 8 * ((int)kmax + 16) * (int)sizeof(__half);
  void* fn = pick_stream3(ntok, (int)policy, (int)math);
  const MatDesc3* gm = reinterpret_cast<const MatDesc3*>(descs.data_ptr<int64_t>());
  const StageDesc3* gs = reinterpret_cast<const StageDesc3*>(stages.data_ptr<int>());
  int nm = (int)descs.size(0), ns = (int)stages.size(0), m = (int)M, its = (int)iters;
  float sc = (float)out_scale;
  long long* pp = prof.numel() > 0 ? reinterpret_cast<long long*>(prof.data_ptr<int64_t>()) : nullptr;
  void* args[] = {&gm, &gs, &nm, &ns, &m, &its, &sc, &pp};
  CUDA_OK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  if (!timed) {
    CUDA_OK(cudaLaunchCooperativeKernel(fn, dim3(sms), dim3(V3_THREADS), args, smem, stream));
    return 0.0;
  }
  EventTimer t(stream);
  t.start();
  CUDA_OK(cudaLaunchCooperativeKernel(fn, dim3(sms), dim3(V3_THREADS), args, smem, stream));
  return t.stop_ms();
}

void run_stream3(torch::Tensor descs, torch::Tensor stages, int64_t M, int64_t kmax, int64_t policy, int64_t math,
                 double out_scale, torch::Tensor prof) {
  launch_stream3(descs, stages, M, 1, kmax, policy, math, out_scale, prof, false);
}

double time_stream3(torch::Tensor descs, torch::Tensor stages, int64_t M, int64_t iters, int64_t kmax, int64_t policy,
                    int64_t math) {
  auto none = torch::empty({0}, descs.options());
  launch_stream3(descs, stages, M, 1, kmax, policy, math, 1.0, none, false);  // warm-up
  return launch_stream3(descs, stages, M, iters, kmax, policy, math, 1.0, none, true);
}

// ======================================================================
// v4: round-robin placement + D-deep prefetch ring
// ======================================================================
// Run 4 (v3) profile: big stages stream at ~225-234 GB/s while computing, but
// ~1.7 us per stage boundary (activation load + barrier) leaves DRAM idle,
// and small stages run on only the first SMs (warp-major task placement put
// 288 tasks on 18 of 48 SMs). v4 changes exactly two things, both measurable:
//   * placement: rr=1 maps task slots round-robin over blocks first, so every
//     stage spreads across all SMs
//   * prefetch depth D: each warp keeps D chunks of weight loads in flight
//     (a register shift ring), reaching across stage boundaries so DRAM keeps
//     streaming through barrier gaps
// Layout, dequant and tables are identical to v3.

struct Meta4 {
  int st, mi, nt, kc;
  bool first, last;
};

struct Cursor4 {
  int s, j, c, mi, nt, kc0, n;
  bool valid;
};

__device__ __forceinline__ void cursor4_decode(Cursor4& cu, const MatDesc3* sm, const StageDesc3* ss, int slot,
                                               int nwarps) {
  decode3(sm, ss[cu.s], slot + cu.j * nwarps, cu.mi, cu.nt, cu.kc0, cu.n);
}

__device__ __forceinline__ void cursor4_init(Cursor4& cu, const MatDesc3* sm, const StageDesc3* ss, int ns, int slot,
                                             int nwarps) {
  cu.s = 0;
  cu.j = 0;
  cu.c = 0;
  while (cu.s < ns && slot >= ss[cu.s].total) ++cu.s;
  cu.valid = cu.s < ns;
  if (cu.valid) cursor4_decode(cu, sm, ss, slot, nwarps);
}

__device__ __forceinline__ void cursor4_next(Cursor4& cu, const MatDesc3* sm, const StageDesc3* ss, int ns, int slot,
                                             int nwarps) {
  if (++cu.c < cu.n) return;
  cu.c = 0;
  ++cu.j;
  while (cu.s < ns && slot + cu.j * nwarps >= ss[cu.s].total) {
    ++cu.s;
    cu.j = 0;
  }
  cu.valid = cu.s < ns;
  if (cu.valid) cursor4_decode(cu, sm, ss, slot, nwarps);
}

__device__ __forceinline__ Meta4 cursor4_meta(const Cursor4& cu) {
  Meta4 m;
  m.st = cu.s;
  m.mi = cu.mi;
  m.nt = cu.nt;
  m.kc = cu.kc0 + cu.c;
  m.first = cu.c == 0;
  m.last = cu.c + 1 >= cu.n;
  return m;
}

template <int NTOK, int D>
__global__ void __launch_bounds__(V3_THREADS, 1) stream4_kernel(const MatDesc3* __restrict__ gm,
                                                                const StageDesc3* __restrict__ gs, int n_mats,
                                                                int n_stages, int M, int iters, float out_scale,
                                                                int rr, long long* __restrict__ prof) {
  __shared__ MatDesc3 sm[V3_MAXMATS];
  __shared__ StageDesc3 ss[V3_MAXSTAGES];
  extern __shared__ __align__(16) __half xs[];
  for (int i = threadIdx.x; i < n_mats; i += blockDim.x) sm[i] = gm[i];
  for (int i = threadIdx.x; i < n_stages; i += blockDim.x) ss[i] = gs[i];
  __syncthreads();
  cg::grid_group grid = cg::this_grid();
  const int lane = threadIdx.x & 31;
  const int lw = threadIdx.x >> 5;
  const int nwarps = (gridDim.x * blockDim.x) >> 5;
  const int slot = rr ? (int)(blockIdx.x + gridDim.x * lw) : (int)(blockIdx.x * (blockDim.x >> 5) + lw);
  const bool rec = prof != nullptr && threadIdx.x == 0;
  if (rec && blockIdx.x == 0) {
    prof[0] = clock64();
    prof[1] = globaltimer();
  }
  for (int it = 0; it < iters; ++it) {
    Cursor4 cu;
    cursor4_init(cu, sm, ss, n_stages, slot, nwarps);
    Chunk3 buf[D];
    Meta4 meta[D];
    bool ok[D];
#pragma unroll
    for (int d = 0; d < D; ++d) {
      ok[d] = cu.valid;
      if (cu.valid) {
        load_chunk3<0>(buf[d], sm[cu.mi], cu.nt, cu.kc0 + cu.c, lane);
        meta[d] = cursor4_meta(cu);
        cursor4_next(cu, sm, ss, n_stages, slot, nwarps);
      }
    }
    float acc[NTOK][4];
    for (int st = 0; st < n_stages; ++st) {
      const long long t0 = clock64();
      const MatDesc3& first = sm[ss[st].m0];
      const int K = first.K, Kp = K + 16;
      load_x3<NTOK>(xs, reinterpret_cast<const __half*>(first.X), M, K, Kp);
      __syncthreads();
      const long long t1 = clock64();
      while (ok[0] && meta[0].st == st) {
        Chunk3 tmp;
        Meta4 tm;
        const bool tv = cu.valid;
        if (tv) {  // keep D chunks in flight: issue the next one before computing
          load_chunk3<0>(tmp, sm[cu.mi], cu.nt, cu.kc0 + cu.c, lane);
          tm = cursor4_meta(cu);
          cursor4_next(cu, sm, ss, n_stages, slot, nwarps);
        }
        if (meta[0].first) {
#pragma unroll
          for (int t = 0; t < NTOK; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
        }
        compute_chunk3<NTOK, 1>(buf[0], acc, xs, Kp, meta[0].kc, lane);
        if (meta[0].last) flush3<NTOK>(acc, sm[meta[0].mi], meta[0].nt, M, lane, out_scale);
#pragma unroll
        for (int d = 0; d + 1 < D; ++d) {
          buf[d] = buf[d + 1];
          meta[d] = meta[d + 1];
          ok[d] = ok[d + 1];
        }
        buf[D - 1] = tmp;
        meta[D - 1] = tm;
        ok[D - 1] = tv;
      }
      if (prof != nullptr && it == 0) {
        __syncthreads();
        if (rec) {
          long long* p = prof + 4 + ((long long)st * gridDim.x + blockIdx.x) * 3;
          p[0] = t0;
          p[1] = t1;
          p[2] = clock64();
        }
      }
      grid.sync();
    }
  }
  if (rec && blockIdx.x == 0) {
    prof[2] = clock64();
    prof[3] = globaltimer();
  }
}

template <int NTOK, int D>
static void* s4fn() {
  return (void*)stream4_kernel<NTOK, D>;
}

static void* pick_stream4(int ntok, int depth) {
  if (ntok == 1) return depth == 1 ? s4fn<1, 1>() : depth == 2 ? s4fn<1, 2>() : s4fn<1, 3>();
  return depth == 1 ? s4fn<2, 1>() : s4fn<2, 2>();
}

static double launch_stream4(torch::Tensor descs, torch::Tensor stages, int64_t M, int64_t iters, int64_t kmax,
                             int64_t depth, int64_t rr, double out_scale, torch::Tensor prof, bool timed) {
  TORCH_CHECK(descs.is_cuda() && descs.scalar_type() == torch::kInt64 && descs.size(1) == 7);
  TORCH_CHECK(stages.is_cuda() && stages.scalar_type() == torch::kInt32 && stages.size(1) == 4);
  TORCH_CHECK(descs.size(0) <= V3_MAXMATS && stages.size(0) <= V3_MAXSTAGES);
  TORCH_CHECK(M >= 1 && M <= 16);
  const int ntok = M <= 8 ? 1 : 2;
  TORCH_CHECK(depth >= 1 && depth <= (ntok == 1 ? 3 : 2), "depth: 1-3 for M <= 8, 1-2 for M > 8");
  auto stream = at::cuda::getCurrentCUDAStream();
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const int smem = ntok * 8 * ((int)kmax + 16) * (int)sizeof(__half);
  void* fn = pick_stream4(ntok, (int)depth);
  const MatDesc3* gm = reinterpret_cast<const MatDesc3*>(descs.data_ptr<int64_t>());
  const StageDesc3* gs = reinterpret_cast<const StageDesc3*>(stages.data_ptr<int>());
  int nm = (int)descs.size(0), ns = (int)stages.size(0), m = (int)M, its = (int)iters, r = (int)rr;
  float sc = (float)out_scale;
  long long* pp = prof.numel() > 0 ? reinterpret_cast<long long*>(prof.data_ptr<int64_t>()) : nullptr;
  void* args[] = {&gm, &gs, &nm, &ns, &m, &its, &sc, &r, &pp};
  CUDA_OK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  if (!timed) {
    CUDA_OK(cudaLaunchCooperativeKernel(fn, dim3(sms), dim3(V3_THREADS), args, smem, stream));
    return 0.0;
  }
  EventTimer t(stream);
  t.start();
  CUDA_OK(cudaLaunchCooperativeKernel(fn, dim3(sms), dim3(V3_THREADS), args, smem, stream));
  return t.stop_ms();
}

void run_stream4(torch::Tensor descs, torch::Tensor stages, int64_t M, int64_t kmax, int64_t depth, int64_t rr,
                 double out_scale, torch::Tensor prof) {
  launch_stream4(descs, stages, M, 1, kmax, depth, rr, out_scale, prof, false);
}

double time_stream4(torch::Tensor descs, torch::Tensor stages, int64_t M, int64_t iters, int64_t kmax, int64_t depth,
                    int64_t rr) {
  auto none = torch::empty({0}, descs.options());
  launch_stream4(descs, stages, M, 1, kmax, depth, rr, 1.0, none, false);  // warm-up
  return launch_stream4(descs, stages, M, iters, kmax, depth, rr, 1.0, none, true);
}
