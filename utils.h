#pragma once
#include <cstdint>

// Encode shared-memory addresses and offsets in 16-byte units.
__device__ inline uint64_t desc_encode(uint64_t value) {
  return (value & 0x3FFFFULL) >> 4;
}

__device__ inline
void tcgen05_mma_f16(int taddr, uint64_t a_desc, uint64_t b_desc,
                    uint32_t i_desc, int accumulate) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.u32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
    "}"
    :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(accumulate)
    : "memory");
}

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cute/arch/cluster_sm90.hpp#L180
// choose one thread from a warp. It returns 1 for that thread and 0 for the others.
__device__ inline
uint32_t elect_sync() {
  uint32_t pred = 0; // each thread has its own result = 0
  asm volatile(
    "{\n\t"
    ".reg .pred %%px;\n\t"
    "elect.sync _|%%px, %1;\n\t"
    "@%%px mov.s32 %0, 1;\n\t"
    "}"
    : "+r"(pred)
    : "r"(0xFFFFFFFF)
  );
  return pred;
}

// init a bariier in shared memory
__device__ inline
void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cutlass/arch/barrier.h#L408
__device__ inline
void mbarrier_wait(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;  // this is optional
  asm volatile(
    "{\n\t"
    ".reg .pred P1;\n\t"
    "LAB_WAIT:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
    "@P1 bra.uni DONE;\n\t"
    "bra.uni LAB_WAIT;\n\t"
    "DONE:\n\t"
    "}"
    :: "r"(mbar_addr), "r"(phase), "r"(ticks)
  );
}

__device__ inline
void tma_2d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr) : "memory");
}

__device__ inline
void tma_3d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z,
                      int mbar_addr) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr)
              : "memory");
}
