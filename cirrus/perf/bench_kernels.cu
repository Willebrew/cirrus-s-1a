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
