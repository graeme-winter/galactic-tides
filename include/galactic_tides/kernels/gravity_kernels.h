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
