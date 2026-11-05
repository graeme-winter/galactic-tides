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

// Tree: one threadgroup per group, one thread per target in it. All threads
// in a threadgroup share the group's interaction list, so the cells are
// staged through threadgroup memory a tile at a time: each thread fetches
// one cell (index, then the 48 byte cell) and after a barrier every thread
// evaluates the whole tile. That turns a chain of dependent loads per
// interaction into one parallel load per tile. Threads beyond the group's
// targets help load and take part in the barriers, then stop. Dispatched
// with threadgroups of at most GT_TREE_TILE threads, at least as many as the
// largest group.
kernel void gt_tree(device const float4 *bodies [[buffer(0)]],
                    device const GTCell *cells [[buffer(1)]],
                    device const uint *cell_list [[buffer(2)]],
                    device const GTRange *leaf_list [[buffer(3)]],
                    device const GTGroup *groups [[buffer(4)]],
                    device float4 *out [[buffer(5)]],
                    constant GTParams &params [[buffer(6)]],
                    uint gid [[threadgroup_position_in_grid]],
                    uint lid [[thread_position_in_threadgroup]],
                    uint tgsize [[threads_per_threadgroup]]) {
  threadgroup float4 tile_cm[GT_TREE_TILE];
  threadgroup float4 tile_q0[GT_TREE_TILE];
  threadgroup float4 tile_q1[GT_TREE_TILE];
  GTParams p = params;
  if (gid >= p.n_groups) return;  // whole threadgroup, so no barrier is left waiting
  GTGroup g = groups[gid];
  uint i = g.begin + lid;
  bool active = i < g.end;
  float4 bi = float4(0.0f);
  if (active) bi = bodies[i];
  GTAcc a = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint base = g.cell_begin; base < g.cell_end; base += tgsize) {
    uint k = base + lid;
    if (k < g.cell_end) {
      GTCell c = cells[cell_list[k]];
      tile_cm[lid] = c.com_mass;
      tile_q0[lid] = c.q0;
      tile_q1[lid] = c.q1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint count = min(tgsize, g.cell_end - base);
    for (uint t = 0; t < count; ++t) {
      GTCell c;
      c.com_mass = tile_cm[t];
      c.q0 = tile_q0[t];
      c.q1 = tile_q1[t];
      a = gt_cell(a, bi.x, bi.y, bi.z, c, p.G, p.eps2, p.quadrupole);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!active) return;  // after the last barrier
  a = gt_group_bodies(a, i, bi, g, bodies, leaf_list, p);
  out[i] = float4(a.ax, a.ay, a.az, a.pot);
}
