#include <cuda.h>
#include <cstdint>
#include <cuda_bf16.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include "profiler.h"
#include "utils.h"

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

constexpr int BLOCK_M = 128;
constexpr int MMA_K = 16;

template <int BLOCK_N, int BLOCK_K, int NUM_STAGES, bool PROFILE>
__global__
__launch_bounds__(TB_SIZE)
void matmul_v5_kernel(
  const __grid_constant__ CUtensorMap A_tmap,
  const __grid_constant__ CUtensorMap B_tmap,
  nv_bfloat16 *C_ptr,
  int M, int N, int K,
  int64_t *profile_data, int num_profile_entries
) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  const int warp_id = tid / WARP_SIZE;
  const bool is_leader = warp_id == 0 && elect_sync();

  const int grid_n = N / BLOCK_N;
  const int bid_m = bid / grid_n;
  const int bid_n = bid % grid_n;

  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  // set up smem
  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int B_size = BLOCK_N * BLOCK_K * sizeof(nv_bfloat16);

  // set up mbarrier and tmem
  // TODO: check if speed is slower if mbar is a variable instead of an array
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ uint64_t tma_mbars[NUM_STAGES];
  __shared__ uint64_t mma_mbars[1];
  __shared__ int tmem_addr[1];  // tmem address is 32-bit
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(tma_mbars));
  const int mma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mma_mbars));

  Profiler profiler;
  if constexpr (PROFILE) {
    if (is_leader) {
      profiler.init(num_profile_entries, profile_data, bid);
      profiler.start(Setup);
    }
  }

  if (is_leader) {
    for (int i = 0; i < NUM_STAGES; i++)
      mbarrier_init(tma_mbar_addr + i * 8, 1);  // only 1 thread issue
    mbarrier_init(mma_mbar_addr, 1);  // only 1 thread issue
    asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
  }
  else if (warp_id == 1) {
    // allocate tmem for output
    const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(addr), "r"(BLOCK_N));
  }

  __syncthreads();  // visible to all threads
  if constexpr (PROFILE) {
    if (is_leader)
      profiler.stop();
  }
  const int taddr = tmem_addr[0];  // this will be 0

  int tma_phase = 0;
  int mma_phase = 0;

  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
  constexpr uint32_t i_desc = (1U << 4U)   // dtype=FP32
                            | (1U << 7U)   // atype=BF16
                            | (1U << 10U)  // btype=BF16
                            | ((uint32_t)BLOCK_N >> 3U << 17U)  // MMA_N
                            | ((uint32_t)BLOCK_M >> 4U << 24U)  // MMA_M
                            ;

  auto load = [&](int iter_k) {
    if (is_leader) {
      if constexpr (PROFILE)
        profiler.start(IssueTMA);
      const int stage_id = iter_k % NUM_STAGES;
      const int mbar_addr = tma_mbar_addr + stage_id * 8;
      const int A_smem = smem + stage_id * (A_size + B_size);
      const int B_smem = A_smem + A_size;

      const int off_k = iter_k * BLOCK_K;
      tma_3d_gmem2smem(A_smem, &A_tmap, 0, off_m, off_k / 64, mbar_addr);
      tma_3d_gmem2smem(B_smem, &B_tmap, 0, off_n, off_k / 64, mbar_addr);
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                  :: "r"(mbar_addr), "r"(A_size + B_size) : "memory");
      if constexpr (PROFILE)
        profiler.stop();
    }
  };

  auto compute = [&](int iter_k) {
    // wait for TMA
    const int stage_id = iter_k % NUM_STAGES;
    const int mbar_addr = tma_mbar_addr + stage_id * 8;
    if constexpr (PROFILE) {
      if (is_leader)
        profiler.start(WaitTMA);
    }
    mbarrier_wait(mbar_addr, tma_phase);
    if constexpr (PROFILE) {
      if (is_leader)
        profiler.stop();
    }
    asm volatile("tcgen05.fence::after_thread_sync;");  // (why) do we need this? from DeepGEMM

    const int A_smem = smem + stage_id * (A_size + B_size);
    const int B_smem = A_smem + A_size;

    // flip TMA phase when we have cycled through all TMA buffers
    if (stage_id == NUM_STAGES - 1)
      tma_phase ^= 1;

    // MMA
    if (is_leader) {
      if constexpr (PROFILE)
        profiler.start(IssueMMA);
      // set up shared memory descriptors for A and B
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor
      // 128-byte swizzling. LBO is implied to be 1.
      auto make_desc = [](int addr) -> uint64_t {
        const int SBO = 8 * 128;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
      };

      // manually unroll 1st iteration to disable accumulation
      {
        tcgen05_mma_f16(taddr, make_desc(A_smem), make_desc(B_smem), i_desc, iter_k);
        for (int k2 = 1; k2 < 64 / MMA_K; k2++) {
          uint64_t a_desc = make_desc(A_smem + k2 * 32);
          uint64_t b_desc = make_desc(B_smem + k2 * 32);
          tcgen05_mma_f16(taddr, a_desc, b_desc, i_desc, 1);
        }
      }
      // k1 selects the (BLOCK_M, 64) tile.
      // k2 selects the (BLOCK_M, 16) tile, whose rows are swizzled.
      for (int k1 = 1; k1 < BLOCK_K / 64; k1++)
        for (int k2 = 0; k2 < 64 / MMA_K; k2++) {
          uint64_t a_desc = make_desc(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
          uint64_t b_desc = make_desc(B_smem + k1 * BLOCK_N * 128 + k2 * 32);
          tcgen05_mma_f16(taddr, a_desc, b_desc, i_desc, 1);
        }
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                  :: "r"(mma_mbar_addr) : "memory");
      if constexpr (PROFILE)
        profiler.stop();
    }
  };

  const int num_iters = K / BLOCK_K;

  // prefetch
  for (int i = 0; i < NUM_STAGES - 1; i++)
    load(i);

  for (int iter_k = 0; iter_k < num_iters - NUM_STAGES + 1; iter_k++) {
    load(iter_k + NUM_STAGES - 1);
    compute(iter_k);

    // wait for MMA
    if constexpr (PROFILE) {
      if (is_leader)
        profiler.start(WaitMMA);
    }
    mbarrier_wait(mma_mbar_addr, mma_phase);
    if constexpr (PROFILE) {
      if (is_leader)
        profiler.stop();
    }
    mma_phase ^= 1;  // flip the phase
  }

  // finish the last few buffers
  for (int iter_k = num_iters - NUM_STAGES + 1; iter_k < num_iters; iter_k++) {
    compute(iter_k);
    if constexpr (PROFILE) {
      if (is_leader)
        profiler.start(WaitMMA);
    }
    mbarrier_wait(mma_mbar_addr, mma_phase);
    if constexpr (PROFILE) {
      if (is_leader)
        profiler.stop();
    }
    mma_phase ^= 1;  // flip the phase
  }

  // PTX doc says we need to add this before tcgen05.ld, after tcgen05.mma
  asm volatile("tcgen05.fence::after_thread_sync;");
  if constexpr (PROFILE) {
    if (is_leader)
      profiler.start(Epilogue);
  }

  // load 8 columns from tmem at a time -> store 16 bytes per thread to smem
  // (still strided though)
  for (int n = 0; n < BLOCK_N / 8; n++) {
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-d
    // Layout D
    float tmp[8];
    const int addr = taddr + ((warp_id * 32) << 16) + (n * 8);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
                  "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
                : "r"(addr));
    asm volatile("tcgen05.wait::ld.sync.aligned;");

    nv_bfloat162 out[4];
    for (int i = 0; i < 4; i++)
      out[i] = __float22bfloat162_rn({tmp[i * 2], tmp[i * 2 + 1]});

    // uncoalesced writes weeee
    nv_bfloat16 *out_ptr = C_ptr + (off_m + tid) * N + (off_n + n * 8);
    reinterpret_cast<int4 *>(out_ptr)[0] = reinterpret_cast<int4 *>(out)[0];
  }
  __syncthreads();  // all threads finish reading data from tmem
  if constexpr (PROFILE) {
    if (is_leader) {
      profiler.stop();
      profiler.flush();
    }
  }
  if (warp_id == 0)  // deallocate tmem
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(BLOCK_N));
}

template <int BLOCK_N, int BLOCK_K, int NUM_STAGES, bool PROFILE>
void matmul_v5_launch(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K, cudaStream_t stream,
  int64_t *profile_data = nullptr, int num_profile_entries = 0
) {
  CUtensorMap A_tmap, B_tmap;

  // input layout for tcgen05: contiguous blocks of (MMA_M, 64)
  // then we perform swizzling within this block
  // 3D tensormap (WIDTH / 64, HEIGHT, 64) : (64, WIDTH, 1)
  auto init_tmap_AB = [&](CUtensorMap *tmap, const nv_bfloat16 *ptr, uint64_t global_height, uint32_t shared_height) {
    constexpr uint32_t rank = 3;
    uint64_t globalDim[rank]       = {64, global_height, (uint64_t)K / 64};
    uint64_t globalStrides[rank-1] = {(uint64_t)K * sizeof(nv_bfloat16), 128};  // in bytes
    uint32_t boxDim[rank]          = {64, shared_height, (uint32_t)BLOCK_K / 64};
    uint32_t elementStrides[rank]  = {1, 1, 1};

    const CUresult err = cuTensorMapEncodeTiled(
      tmap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      rank,
      (void *)ptr,
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    TORCH_CHECK(err == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(err));
  };
  init_tmap_AB(&A_tmap, A_ptr, M, BLOCK_M);
  init_tmap_AB(&B_tmap, B_ptr, N, BLOCK_N);

  int grid = (M / BLOCK_M) * (N / BLOCK_N);
  int size_AB = (BLOCK_M + BLOCK_N) * BLOCK_K * NUM_STAGES;
  int smem_size = size_AB * sizeof(nv_bfloat16);

  auto this_kernel = matmul_v5_kernel<BLOCK_N, BLOCK_K, NUM_STAGES, PROFILE>;
  if (smem_size > 48'000)
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  this_kernel<<<grid, TB_SIZE, smem_size, stream>>>(
    A_tmap, B_tmap, C_ptr, M, N, K, profile_data, num_profile_entries);
}

void matmul_v5_cuda(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K, cudaStream_t stream
) {
  matmul_v5_launch<256, 128, 2, false>(A_ptr, B_ptr, C_ptr, M, N, K, stream);
}

void profile_matmul_v5_cuda(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
  nv_bfloat16 *C_ptr,
  int M, int N, int K, cudaStream_t stream,
  int64_t *profile_data, int num_profile_entries
) {
  matmul_v5_launch<256, 128, 2, true>(
    A_ptr, B_ptr, C_ptr, M, N, K, stream, profile_data, num_profile_entries);
}

void check_matmul_inputs(const at::Tensor& A, const at::Tensor& B, const char* version) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.device() == B.device(),
              "A and B must be CUDA tensors on the same device");
  TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16,
              version, " requires BF16 inputs");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(0),
              "Expected A[M, K] and B[K, N]");
  TORCH_CHECK(A.is_contiguous() && B.stride(0) == 1 && B.stride(1) == B.size(0),
              "A must be contiguous; B must be a transposed contiguous [N, K] tensor");
  TORCH_CHECK(A.size(0) > 0 && B.size(1) > 0 && A.size(1) > 0 &&
              A.size(0) % BLOCK_M == 0 && B.size(1) % 256 == 0 && A.size(1) % 128 == 0,
              version, " requires M a multiple of 128 and N/K multiples of 256/128");
}

at::Tensor matmul_v5(const at::Tensor& A, const at::Tensor& B) {
  check_matmul_inputs(A, B, "v5");
  const c10::cuda::CUDAGuard guard(A.device());
  auto C = at::empty({A.size(0), B.size(1)}, A.options());
  matmul_v5_cuda(
    reinterpret_cast<const nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
    reinterpret_cast<const nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
    reinterpret_cast<nv_bfloat16*>(C.data_ptr<at::BFloat16>()),
    A.size(0), B.size(1), A.size(1), c10::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return C;
}

at::Tensor profile_matmul_v5(
  const at::Tensor& A, const at::Tensor& B, const at::Tensor& profiler, int64_t num_profile_entries
) {
  check_matmul_inputs(A, B, "v5");
  TORCH_CHECK(profiler.is_cuda() && profiler.device() == A.device() && profiler.scalar_type() == at::kLong,
              "profiler must be an int64 CUDA tensor on the input device");
  const auto num_blocks = (A.size(0) / BLOCK_M) * (B.size(1) / 256);
  TORCH_CHECK(profiler.dim() == 2 && profiler.size(0) == num_blocks &&
              profiler.size(1) >= 1 + num_profile_entries * 4,
              "profiler has an incompatible shape");
  const c10::cuda::CUDAGuard guard(A.device());
  auto C = at::empty({A.size(0), B.size(1)}, A.options());
  profile_matmul_v5_cuda(
    reinterpret_cast<const nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
    reinterpret_cast<const nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
    reinterpret_cast<nv_bfloat16*>(C.data_ptr<at::BFloat16>()),
    A.size(0), B.size(1), A.size(1), c10::cuda::getCurrentCUDAStream(),
    profiler.data_ptr<int64_t>(), num_profile_entries);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return C;
}

TORCH_LIBRARY_FRAGMENT(my_matmul, m) {
  m.def("matmul_v5(Tensor A, Tensor B) -> Tensor");
  m.def("profile_matmul_v5(Tensor A, Tensor B, Tensor profiler, int num_profile_entries) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_matmul, CUDA, m) {
  m.impl("matmul_v5", &matmul_v5);
  m.impl("profile_matmul_v5", &profile_matmul_v5);
}
