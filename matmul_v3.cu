#include <cuda.h>      // CUtensorMap, cuTensorMapEncodeTiled, enums
#include <cstdint>     // uint64_t, uint32_t
#include <cuda_bf16.h>

#include "utils.h"

namespace {

constexpr int WARP_SIZE = 32;

constexpr int NUM_WARPS = 4;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE; // threads per block

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 256;

constexpr int MMA_K = 16;

// to build a TMA descriptor for a contiguous row major bf16 matrix
void init_tmap_2d_simple(
  CUtensorMap *tmap, // output descriptor
  const nv_bfloat16 *ptr, // matrix base address
  uint64_t global_height, uint64_t global_width, // matrix shape
  uint32_t shared_height, uint32_t shared_width, // transfer tile shape
  CUtensorMapSwizzle swizzle
) {
  constexpr uint32_t rank = 2; // 2d
  uint64_t globalDim[rank]       = {global_width, global_height}; // tma order dimensions fastest varying first (columns, rows)
  uint64_t globalStrides[rank-1] = {global_width * sizeof(nv_bfloat16)};  // in bytes
  uint32_t boxDim[rank]          = {shared_width, shared_height};
  uint32_t elementStrides[rank]  = {1, 1}; // consecutive elements

  // global strides is the distance between each rows
  // element strides for INTERLEAVE NONE, the first 1 that denotes the column is ignored, so the second 1 denotes the row.
  // element strides specify step size (1 means every row ie 0, 1, 2, 3.  2 means 0, 2, 4, 6)

  cuTensorMapEncodeTiled(
    tmap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    rank,
    (void *)ptr,
    globalDim,
    globalStrides,
    boxDim,
    elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    swizzle,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
}

// Reinterpret a row-major [height, width] matrix as [height, width / 64, 64].
// The innermost dimension is 64 BF16 elements, i.e. a 128-byte TMA slice.
void init_tmap_3d_128b(
  CUtensorMap *tmap,
  const nv_bfloat16 *ptr,
  uint64_t global_height, uint64_t global_width,
  uint32_t shared_height
) {
  constexpr uint32_t rank = 3;
  constexpr uint32_t K_SLICE = 64;
  uint64_t globalDim[rank] = {K_SLICE, global_width / K_SLICE, global_height};
  uint64_t globalStrides[rank - 1] = {
    K_SLICE * sizeof(nv_bfloat16), global_width * sizeof(nv_bfloat16)};
  uint32_t boxDim[rank] = {K_SLICE, 1, shared_height};
  uint32_t elementStrides[rank] = {1, 1, 1};

  cuTensorMapEncodeTiled(
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
}

template <bool USE_3D_TMA>
__global__
__launch_bounds__(TB_SIZE)
void matmul_kernel(
  const __grid_constant__ CUtensorMap A_tmap,
  const __grid_constant__ CUtensorMap B_tmap,
  nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  const int tid = threadIdx.x;

  const int warp_id = tid / WARP_SIZE;

  // v0 uses a 2D output-tile grid.
  const int bid_m = blockIdx.y;
  const int bid_n = blockIdx.x;

  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  // set up smem
  extern __shared__ __align__(1024) char smem[];
  const int A_smem = static_cast<int>(__cvta_generic_to_shared(smem)); // convert cuda ptr to smem adreess for inline ptx
  const int B_smem = A_smem + BLOCK_M * BLOCK_K * sizeof(nv_bfloat16); // B starts after As BLOCk_M x BLOCK_K bf16 elements

  // set up mbarrier and tmem
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ uint64_t mbar;
  __shared__ int tmem_addr;  // tmem address is 32-bit
  const int mbar_addr = static_cast<int>(__cvta_generic_to_shared(&mbar));

  if (warp_id == 0 && elect_sync()) { // can we not simply use tid == 0?
    mbarrier_init(mbar_addr, 1);  // 1 means one arrival is required
    asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
  }
  else if (warp_id == 1) {
    // allocate tmem for output
    const int addr = static_cast<int>(__cvta_generic_to_shared(&tmem_addr));
    // tmem has fixed size of 128 rows, so we need tell it how many columns to allocate
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(addr), "r"(BLOCK_N)); 
  }

  __syncthreads();  // visible to all threads
  const int taddr = tmem_addr;  // allocated tensor-memory address

  int phase = 0;

  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
  constexpr uint32_t i_desc = (1U << 4U)   // dtype=FP32
                            | (1U << 7U)   // atype=BF16
                            | (1U << 10U)  // btype=BF16
                            | ((uint32_t)BLOCK_N >> 3U << 17U)  // MMA_N
                            | ((uint32_t)BLOCK_M >> 4U << 24U)  // MMA_M
                            ;

  const int num_iters = K / BLOCK_K;
  for (int iter_k = 0; iter_k < num_iters; iter_k++) {
    // load
    if (warp_id == 0 && elect_sync()) {
      // (BLOCK_K / 8) issues for each A and B. this is because each TMA call copies 8 columns according to our tensor map
      for (int k = 0; k < BLOCK_K / 64; k++) {
        const int off_k = iter_k * BLOCK_K + k * 64;
        // each A copy transfers block M rows * 8 bf16 = M row * 16 bytes
        // we do off_k and off_m here because in descriptor we have (global width, global height)
        if constexpr (USE_3D_TMA) {
          const int k_slice = iter_k * (BLOCK_K / 64) + k;
          tma_3d_gmem2smem(A_smem + k * BLOCK_M * 128, &A_tmap, 0, k_slice, off_m, mbar_addr);
          tma_3d_gmem2smem(B_smem + k * BLOCK_N * 128, &B_tmap, 0, k_slice, off_n, mbar_addr);
        } else {
          tma_2d_gmem2smem(A_smem + k * BLOCK_M * 128, &A_tmap, off_k, off_m, mbar_addr);
          tma_2d_gmem2smem(B_smem + k * BLOCK_N * 128, &B_tmap, off_k, off_n, mbar_addr);
        }
      }

      constexpr int cp_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                  :: "r"(mbar_addr), "r"(cp_size) : "memory");
    }

    // wait for TMA
    mbarrier_wait(mbar_addr, phase);
    asm volatile("tcgen05.fence::after_thread_sync;");  // (why) do we need this? from DeepGEMM
    phase ^= 1;  // flip the phase

    // MMA
    if (warp_id == 0 && elect_sync()) {
        auto make_desc = [](int addr) -> uint64_t {
            const int SBO = 8 * 128;  // size of the 8x128B tile
            return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
        };

      // manually unroll 1st iteration to disable accumulation
      tcgen05_mma_f16(taddr, make_desc(A_smem), make_desc(B_smem), i_desc, iter_k);
        for (int k2 = 1; k2 < 64 / MMA_K; k2++) {
        uint64_t a_desc = make_desc(A_smem + k2 * 32);
        uint64_t b_desc = make_desc(B_smem + k2 * 32);
        tcgen05_mma_f16(taddr, a_desc, b_desc, i_desc, 1);
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
                  :: "r"(mbar_addr) : "memory");
    }

    // wait for MMA
    mbarrier_wait(mbar_addr, phase);
    phase ^= 1;  // flip the phase

  } // Finish every K chunk before storing and deallocating the result.

    // PTX doc says we need to add this before tcgen05.ld, after tcgen05.mma
    asm volatile("tcgen05.fence::after_thread_sync;");

    // load 8 columns from tmem at a time -> store 16 bytes per thread to smem
    // (still strided though)
    for (int n = 0; n < BLOCK_N / 8; n++) { // block n is the column of the output tile
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-d
        // Layout D
        float tmp[8];
        const int addr = taddr + ((warp_id * 32) << 16) + (n * 8); // warp_id * 32 is the warp’s starting row and n * 8 is starting column
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
                    "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
                    : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        nv_bfloat162 out[4];
        for (int i = 0; i < 4; i++)
        out[i] = __float22bfloat162_rn({tmp[i * 2], tmp[i * 2 + 1]});

        // uncoalesced writes 
        // C[row, col] = Cptr + row * N + col
        nv_bfloat16 *out_ptr = C_ptr + (off_m + tid) * N + (off_n + n * 8);
        reinterpret_cast<int4 *>(out_ptr)[0] = reinterpret_cast<int4 *>(out)[0];
    }
    __syncthreads();  // all threads finish reading data from tmem
    if(warp_id == 0)  // deallocate tmem
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(BLOCK_N));
}


template <bool USE_3D_TMA>
void matmul(
    const nv_bfloat16 *A_ptr,
    const nv_bfloat16 *B_ptr,
    nv_bfloat16 *C_ptr,
    int M, int N, int K, cudaStream_t stream = nullptr
){
    CUtensorMap A_tmap, B_tmap;

    if constexpr (USE_3D_TMA) {
      init_tmap_3d_128b(&A_tmap, A_ptr, M, K, BLOCK_M);
      init_tmap_3d_128b(&B_tmap, B_ptr, N, K, BLOCK_N);
    } else {
      init_tmap_2d_simple(&A_tmap, A_ptr, M, K, BLOCK_M, 64, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
      init_tmap_2d_simple(&B_tmap, B_ptr, N, K, BLOCK_N, 64, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    }

    dim3 grid(N / BLOCK_N, M / BLOCK_M);
    int size_AB = (BLOCK_M + BLOCK_N) * BLOCK_K; // total bf16 elements you need in smem for one A tile and one B tile
    int smem_size = size_AB * sizeof(nv_bfloat16);

  auto this_kernel = matmul_kernel<USE_3D_TMA>;
    if (smem_size > 48'000)
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    this_kernel<<<grid, TB_SIZE, smem_size, stream>>>(A_tmap, B_tmap, C_ptr, M, N, K);
}

}  // namespace

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

at::Tensor matmul_v3(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.device() == B.device(),
                "A and B must be on the same CUDA device");
    TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16,
                "A and B must be BF16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(0),
                "Expected A[M, K] and B[K, N]");
    TORCH_CHECK(A.is_contiguous() && B.stride(0) == 1 && B.stride(1) == B.size(0),
                "A must be contiguous; B must be a transposed contiguous [N, K] tensor");
    const auto M = A.size(0), N = B.size(1), K = A.size(1);
    TORCH_CHECK(M > 0 && N > 0 && K > 0 && M % BLOCK_M == 0 &&
                N % BLOCK_N == 0 && K % BLOCK_K == 0,
                "v3 requires positive M a multiple of 128 and N/K multiples of 256");
    const c10::cuda::CUDAGuard guard(A.device());
    auto C = at::empty({M, N}, A.options());
    matmul<false>(
        reinterpret_cast<const nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16*>(C.data_ptr<at::BFloat16>()),
        M, N, K, c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

at::Tensor matmul_v4(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.device() == B.device(),
                "A and B must be on the same CUDA device");
    TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16,
                "A and B must be BF16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(0),
                "Expected A[M, K] and B[K, N]");
    TORCH_CHECK(A.is_contiguous() && B.stride(0) == 1 && B.stride(1) == B.size(0),
                "A must be contiguous; B must be a transposed contiguous [N, K] tensor");
    const auto M = A.size(0), N = B.size(1), K = A.size(1);
    TORCH_CHECK(M > 0 && N > 0 && K > 0 && M % BLOCK_M == 0 &&
                N % BLOCK_N == 0 && K % BLOCK_K == 0,
                "v4 requires positive M a multiple of 128 and N/K multiples of 256");
    const c10::cuda::CUDAGuard guard(A.device());
    auto C = at::empty({M, N}, A.options());
    matmul<true>(
        reinterpret_cast<const nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16*>(C.data_ptr<at::BFloat16>()),
        M, N, K, c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

TORCH_LIBRARY_FRAGMENT(my_matmul, m) {
    m.def("matmul_v3(Tensor A, Tensor B) -> Tensor");
    m.def("matmul_v4(Tensor A, Tensor B) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_matmul, CUDA, m) {
    m.impl("matmul_v3", &matmul_v3);
    m.impl("matmul_v4", &matmul_v4);
}
