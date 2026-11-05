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

// Tree: one group per SIMD group, several SIMD groups per threadgroup. Lanes
// share the group's list entries with simd_shuffle: each lane fetches one
// entry (an index, then the cell or body it names) into registers, then every
// lane evaluates all of them, taking each in turn from the lane that fetched
// it. No threadgroup memory and no barriers. Groups must have at most
// GT_SIMD_WIDTH targets and the pipeline's SIMD width must be GT_SIMD_WIDTH;
// every lane of a SIMD group runs every loop iteration (lanes past the
// group's targets still fetch), as simd_shuffle requires.
kernel void gt_tree(device const float4 *bodies [[buffer(0)]],
                    device const GTCell *cells [[buffer(1)]],
                    device const uint *cell_list [[buffer(2)]],
                    device const uint *body_list [[buffer(3)]],
                    device const GTGroup *groups [[buffer(4)]],
                    device float4 *out [[buffer(5)]],
                    constant GTParams &params [[buffer(6)]],
                    uint gid [[threadgroup_position_in_grid]],
                    uint simd [[simdgroup_index_in_threadgroup]],
                    uint simds [[simdgroups_per_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  GTParams p = params;
  uint group = gid * simds + simd;
  if (group >= p.n_groups) return;  // the whole SIMD group leaves together
  GTGroup g = groups[group];
  uint i = g.begin + lane;
  bool active = i < g.end;
  float4 bi = float4(0.0f);
  if (active) bi = bodies[i];
  GTAcc a = {0.0f, 0.0f, 0.0f, 0.0f};

  for (uint base = g.cell_begin; base < g.cell_end; base += GT_SIMD_WIDTH) {
    uint k = base + lane;
    float4 cm = float4(0.0f);
    float4 q0 = float4(0.0f);
    float4 q1 = float4(0.0f);
    if (k < g.cell_end) {
      GTCell c = cells[cell_list[k]];
      cm = c.com_mass;
      q0 = c.q0;
      q1 = c.q1;
    }
    uint count = min(GT_SIMD_WIDTH, g.cell_end - base);
    for (uint t = 0; t < count; ++t) {
      GTCell c;
      c.com_mass = simd_shuffle(cm, t);
      c.q0 = simd_shuffle(q0, t);
      c.q1 = simd_shuffle(q1, t);
      a = gt_cell(a, bi.x, bi.y, bi.z, c, p.G, p.eps2, p.quadrupole);
    }
  }

  for (uint base = g.body_begin; base < g.body_end; base += GT_SIMD_WIDTH) {
    uint k = base + lane;
    float4 b = float4(0.0f);
    uint j = 0xffffffffu;
    if (k < g.body_end) {
      j = body_list[k];
      b = bodies[j];
    }
    uint count = min(GT_SIMD_WIDTH, g.body_end - base);
    for (uint t = 0; t < count; ++t) {
      float4 bj = simd_shuffle(b, t);
      uint jt = simd_shuffle(j, t);
      if (jt != i) a = gt_body(a, bi.x, bi.y, bi.z, bj, p.G, p.eps2);
    }
  }

  if (active) out[i] = float4(a.ax, a.ay, a.az, a.pot);
}
