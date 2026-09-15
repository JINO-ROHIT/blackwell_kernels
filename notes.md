### main blackwell features

the main features in data center blackwell gpus are tcgen05 instructions, tensor memory where the MMA's result is stored on new, dedicated hardware - tensor memory. This breaks the WGMMA’s dependency on registers.


1. tensor memory (not the same as TMA that you have in hopper btw)

this came up because of the issues in hopper where MMA and WGMMA store results in registers of each thread, but this also means many other instructions and operations depend on these registers which leads to a lot of contention and harms performance, as operands are spilled into more expensive memory regions like local memory which increases the latency of fetching these operands for operations. With Blackwell datacenter GPUs like the B200, MMA results are now accumulated in a new region of memory called tensor memory, instead of registers.


the tensor memory region is per SM, with tensor memory being organized as a 2d matrix that is 128 lanes (rows) and 512 columns in size. each cell within this matrix is 32 bits.

so the total memory capacity is = 128 * 512 * 4 bytes = 256 kb per SM

unlike other memory regions, tensor memory requires an entire warpgroup (4 warps) for full access of all lanes and columns, also the warpgroup must start at a warpgroup aligned index (multiple of 4). with each warp in the warpgroup being responsible for accessing 32 lanes. Therefore an entire warpgroup is required for the epilogue stage of matrix multiplication for writing computed values from tensor memory back to HBM. tmem must also be deallocated by the kernel explicitly, as it is managed by the programmer, unlike smem where it will be automatically deallocated.

2. cluster launch control

this gives CTAs the ability to cancel queued CTAs that haven't begun executing yet and steal their work by taking the cancelled CTA's index. Unlike non-CLC persistent kernels that launch as many CTAs as there are SMs, CLC kernels launch as many CTAs as there are output tiles. This work stealing approach allows for CTAs to dynamically receive the next tile to work on as opposed to static assignment. Overall this improves load balancing as CTA run-times can exhibit variability so CLC can dynamically adjust for such variability via scheduling.

3. 2-CTA MMA

blackwell tensor cores can be used with the new tcgen05.mma PTX instruction. One new feature of tcgen05.mma is that alongside the new cta_group operand, it enables larger MMA shapes that span across 2-CTAs within a thread block cluster.


kernel 1 explanation

1. init_tmap_2d_simple - this is an instruction that describe where the matrix lives in global memory and what rectangular region a TMA copy should load.

```
init_tmap_2d_simple(
      &B_tmap,       // Write the descriptor here
      B_ptr,         // Base address of B's data in GPU global memory
      N, K,          // Physical layout: N rows, K contiguous elements per row
      BLOCK_N, 8,    // Each copy loads BLOCK_N rows × 8 columns
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE
  );
```

later we use this instruction to do the copy

```
 tma_2d_gmem2smem(
      destination, &B_tmap,
      off_k, off_n,    // Starting physical column and row
      mbar_addr
  );
```

more inline comments within the kernel itself.