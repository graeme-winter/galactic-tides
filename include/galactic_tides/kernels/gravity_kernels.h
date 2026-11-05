// Gravity kernels shared between Metal and C++.
//
// This header is written in the common subset of the Metal Shading Language
// and C++17. On the GPU it is prepended to gravity_kernels.metal and compiled
// at run time; on the CPU it is an ordinary header, so the exact arithmetic
// the GPU runs is also run (and tested) on any platform.
//
// Rules for editing, so that both compilers stay happy:
//   * single precision only, and every literal needs an f suffix (Apple GPUs
//     have no double; 2.5 rather than 2.5f is a compile error in Metal);
//   * buffer pointers carry GT_DEVICE; no references, no std::, no recursion;
//   * structs contain only float, uint and float4, so the host and Metal
//     layouts agree (checked by static_assert on the C++ side);
//   * copy structs out of device memory into locals before passing them on.

#ifndef GT_GRAVITY_KERNELS_H
#define GT_GRAVITY_KERNELS_H

#ifdef __METAL_VERSION__

#include <metal_stdlib>
using namespace metal;
#define GT_DEVICE device
#define GT_BEGIN_NAMESPACE
#define GT_END_NAMESPACE

#else  // C++

#include <cmath>
#define GT_DEVICE
#define GT_BEGIN_NAMESPACE namespace galactic_tides { namespace kern {
#define GT_END_NAMESPACE } }

GT_BEGIN_NAMESPACE
typedef unsigned int uint;
struct alignas(16) float4 {
  float x, y, z, w;
};
inline float rsqrt(float v) { return 1.0f / std::sqrt(v); }
GT_END_NAMESPACE

#endif

GT_BEGIN_NAMESPACE

// Run parameters, passed by value (setBytes on the Metal side).
struct GTParams {
  uint n;           // number of bodies
  uint n_groups;    // tree kernel: number of groups
  float G;
  float eps2;       // Plummer softening squared
  uint quadrupole;  // tree kernel: 1 to include cell quadrupoles
  uint pad0;
  uint pad1;
  uint pad2;
};

// Accumulated acceleration and potential for one target.
struct GTAcc {
  float ax;
  float ay;
  float az;
  float pot;
};

// One softened body-body interaction. Bodies are (x, y, z, mass).
inline GTAcc gt_body(GTAcc a, float px, float py, float pz, float4 b, float G, float eps2) {
  float dx = px - b.x;
  float dy = py - b.y;
  float dz = pz - b.z;
  float inv = rsqrt(dx * dx + dy * dy + dz * dz + eps2);
  float gm = G * b.w * inv;
  float gm3 = gm * inv * inv;
  a.ax -= gm3 * dx;
  a.ay -= gm3 * dy;
  a.az -= gm3 * dz;
  a.pot -= gm;
  return a;
}

// A tree cell as the kernels see it: centre of mass and mass, then the
// traceless quadrupole (xx, yy, zz, xy) and (xz, yz, -, -).
struct GTCell {
  float4 com_mass;
  float4 q0;
  float4 q1;
};

// A run of consecutive bodies (a tree leaf), in tree order.
struct GTRange {
  uint begin;
  uint end;
};

// A group of targets that share one interaction list: bodies [begin, end)
// in tree order, cells cell_list[cell_begin, cell_end) and body ranges
// leaf_list[leaf_begin, leaf_end).
struct GTGroup {
  uint begin;
  uint end;
  uint cell_begin;
  uint cell_end;
  uint leaf_begin;
  uint leaf_end;
  uint pad0;
  uint pad1;
};

// Monopole plus (optionally) quadrupole of an accepted cell, Plummer
// softened as on the CPU tree.
inline GTAcc gt_cell(GTAcc a, float px, float py, float pz, GTCell c, float G, float eps2,
                     uint quadrupole) {
  float dx = px - c.com_mass.x;
  float dy = py - c.com_mass.y;
  float dz = pz - c.com_mass.z;
  float inv = rsqrt(dx * dx + dy * dy + dz * dz + eps2);
  float inv2 = inv * inv;
  float inv3 = inv * inv2;
  float gm = G * c.com_mass.w;
  a.ax -= gm * inv3 * dx;
  a.ay -= gm * inv3 * dy;
  a.az -= gm * inv3 * dz;
  a.pot -= gm * inv;
  if (quadrupole != 0u) {
    float qx = c.q0.x * dx + c.q0.w * dy + c.q1.x * dz;
    float qy = c.q0.w * dx + c.q0.y * dy + c.q1.y * dz;
    float qz = c.q1.x * dx + c.q1.y * dy + c.q0.z * dz;
    float rqr = dx * qx + dy * qy + dz * qz;
    float inv5 = inv3 * inv2;
    float inv7 = inv5 * inv2;
    a.ax += G * (qx * inv5 - 2.5f * rqr * dx * inv7);
    a.ay += G * (qy * inv5 - 2.5f * rqr * dy * inv7);
    a.az += G * (qz * inv5 - 2.5f * rqr * dz * inv7);
    a.pot -= 0.5f * G * rqr * inv5;
  }
  return a;
}

// Target i (tree order) of group g: every cell on the group's list, then
// every body of every leaf on it, skipping itself.
inline GTAcc gt_group_accel(uint i, GTGroup g, GT_DEVICE const float4 *bodies,
                            GT_DEVICE const GTCell *cells, GT_DEVICE const uint *cell_list,
                            GT_DEVICE const GTRange *leaf_list, GTParams p) {
  float4 bi = bodies[i];
  GTAcc a = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint k = g.cell_begin; k < g.cell_end; ++k) {
    GTCell c = cells[cell_list[k]];
    a = gt_cell(a, bi.x, bi.y, bi.z, c, p.G, p.eps2, p.quadrupole);
  }
  for (uint k = g.leaf_begin; k < g.leaf_end; ++k) {
    GTRange r = leaf_list[k];
    for (uint j = r.begin; j < r.end; ++j) {
      if (j == i) continue;
      float4 bj = bodies[j];
      a = gt_body(a, bi.x, bi.y, bi.z, bj, p.G, p.eps2);
    }
  }
  return a;
}

// Brute force: target i against every other body.
inline GTAcc gt_direct_accel(uint i, GT_DEVICE const float4 *bodies, GTParams p) {
  float4 bi = bodies[i];
  GTAcc a = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint j = 0; j < p.n; ++j) {
    if (j == i) continue;
    float4 bj = bodies[j];
    a = gt_body(a, bi.x, bi.y, bi.z, bj, p.G, p.eps2);
  }
  return a;
}

GT_END_NAMESPACE

#endif  // GT_GRAVITY_KERNELS_H
