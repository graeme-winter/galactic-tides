#pragma once
// Runners for the Metal kernels compiled as C++ (metal_mock_kernels.cpp).
// Arrays are passed untyped because that translation unit sees the kernel
// structs in their Metal form; the layouts are identical (static_assert on
// both sides).

namespace mock {
void run_direct(const void *bodies, const void *params, unsigned threads_per_group, void *out);
void run_tree(const void *bodies, const void *cells, const void *cell_list, const void *leaf_list,
              const void *groups, const void *params, unsigned threads_per_group, void *out);
}  // namespace mock
