#pragma once
// Runners for the Metal kernels compiled as C++ (metal_mock_kernels.cpp).
// Arrays are passed untyped because that translation unit sees the kernel
// structs in their Metal form; the layouts are identical (static_assert on
// both sides).

namespace mock {
void run_direct(const void *bodies, const void *params, unsigned threads_per_group, void *out);
void run_tree(const void *bodies, const void *cells, const void *cell_list, const void *body_list,
              const void *groups, const void *params, unsigned threads_per_group, void *out);
// gt_tree_simd: simds_per_threadgroup groups per threadgroup, one per SIMD
// group of 32 lanes (each lane its own thread; simd_shuffle is emulated).
void run_tree_simd(const void *bodies, const void *cells, const void *cell_list,
                   const void *body_list, const void *groups, const void *params,
                   unsigned simds_per_threadgroup, void *out);
// True if any thread left a kernel while others still had barriers (or
// shuffles) to reach.
bool barrier_violation();
}  // namespace mock
