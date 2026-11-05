// Metal compute kernels. At run time this file is appended to
// include/galactic_tides/kernels/gravity_kernels.h (which holds all of the
// arithmetic, shared with the CPU) and compiled with newLibraryWithSource.
// For an offline syntax check on a Mac with Xcode:
//   cmake --build build --target metal_check

#ifndef GT_GRAVITY_KERNELS_H
#include "gravity_kernels.h"
#endif

// Brute force, one thread per target. Each threadgroup stages a tile of
// bodies in threadgroup memory, so every body is read from device memory
// once per threadgroup rather than once per thread. Dispatched with uniform
// threadgroups of at most GT_TILE threads; threads past the end still help
// load tiles and take part in the barriers.
#define GT_TILE 256u

kernel void gt_direct(device const float4 *bodies [[buffer(0)]],
                      device float4 *out [[buffer(1)]],
                      constant GTParams &params [[buffer(2)]],
                      uint i [[thread_position_in_grid]],
                      uint lid [[thread_position_in_threadgroup]],
                      uint tgsize [[threads_per_threadgroup]]) {
  threadgroup float4 tile[GT_TILE];
  GTParams p = params;
  bool active = i < p.n;
  float4 bi = float4(0.0f);
  if (active) bi = bodies[i];
  GTAcc a = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint base = 0; base < p.n; base += tgsize) {
    uint j = base + lid;
    float4 bj = float4(0.0f);
    if (j < p.n) bj = bodies[j];
    tile[lid] = bj;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint count = min(tgsize, p.n - base);
    for (uint k = 0; k < count; ++k) {
      if (base + k != i) a = gt_body(a, bi.x, bi.y, bi.z, tile[k], p.G, p.eps2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (active) out[i] = float4(a.ax, a.ay, a.az, a.pot);
}

// Tree: one threadgroup per group, one thread per target in it. Every thread
// in a threadgroup walks the same interaction list, so list and cell reads
// are uniform across the SIMD group. Dispatched with threadgroups of
// max_group threads; groups smaller than that leave threads idle.
kernel void gt_tree(device const float4 *bodies [[buffer(0)]],
                    device const GTCell *cells [[buffer(1)]],
                    device const uint *cell_list [[buffer(2)]],
                    device const GTRange *leaf_list [[buffer(3)]],
                    device const GTGroup *groups [[buffer(4)]],
                    device float4 *out [[buffer(5)]],
                    constant GTParams &params [[buffer(6)]],
                    uint gid [[threadgroup_position_in_grid]],
                    uint lid [[thread_position_in_threadgroup]]) {
  GTParams p = params;
  if (gid >= p.n_groups) return;
  GTGroup g = groups[gid];
  uint i = g.begin + lid;
  if (i >= g.end) return;
  GTAcc a = gt_group_accel(i, g, bodies, cells, cell_list, leaf_list, p);
  out[i] = float4(a.ax, a.ay, a.az, a.pot);
}
