// The Metal kernel source, compiled as C++ and run with GPU-style threads
// (see metal_mock_kernels.cpp), against the CPU paths and double precision.

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "check.h"
#include "metal_mock_kernels.h"
#include "galactic_tides/gravity.h"
#include "galactic_tides/gravity_backends.h"
#include "galactic_tides/group_lists.h"

using namespace galactic_tides;

namespace {

Particles make_system(std::size_t n, unsigned seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  Particles p;
  for (std::size_t i = 0; i < n; ++i) {
    const double x = std::sqrt(u(rng) * 0.95), r = x / (1 - x);
    const double cz = 2 * u(rng) - 1, ph = 2 * M_PI * u(rng), s = std::sqrt(1 - cz * cz);
    const Vec3 off = (i % 7 == 0) ? Vec3(3, 1, 0) : Vec3();
    p.push_back(off + r * Vec3(s * std::cos(ph), s * std::sin(ph), cz), Vec3(), 1.0 / n, i,
                ParticleType::Halo);
  }
  return p;
}

double max_rel(const std::vector<kern::float4> &a, const std::vector<kern::float4> &b) {
  double worst = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = std::hypot(a[i].x - b[i].x, a[i].y - b[i].y, a[i].z - b[i].z);
    const double s = std::hypot(b[i].x, b[i].y, b[i].z);
    worst = std::max(worst, d / s);
    worst = std::max(worst, std::fabs(double(a[i].w) / b[i].w - 1));
  }
  return worst;
}

}  // namespace

int main() {
  // Odd sizes so the last threadgroup is partial.
  const std::size_t n = 613;
  Particles ref = make_system(n, 3);
  GravityParams gp;
  gp.eps = 0.02;
  compute_gravity_direct(ref, gp);

  // ---- tiled brute force, several threadgroup sizes ---------------------------
  std::vector<kern::float4> bodies;
  pack_bodies(ref.pos, ref.mass, Vec3(), bodies);
  const kern::GTParams kp = make_kernel_params(gp, n);
  std::vector<kern::float4> emulated(n);
  for (kern::uint i = 0; i < n; ++i) {
    const kern::GTAcc a = kern::gt_direct_accel(i, bodies.data(), kp);
    emulated[i] = kern::float4{a.ax, a.ay, a.az, a.pot};
  }
  for (unsigned tg : {1u, 32u, 64u, 256u}) {
    std::vector<kern::float4> out(n, kern::float4{NAN, NAN, NAN, NAN});
    mock::run_direct(bodies.data(), &kp, tg, out.data());
    const double d = max_rel(out, emulated);
    std::cout << "gt_direct, threadgroups of " << tg << ": max relative difference " << d
              << "\n";
    CHECK(d < 1e-4);  // summation order differs, nothing else should
  }

  // ---- tree kernel on grouped lists -------------------------------------------
  for (bool pack : {false, true}) {
    for (int gs : {8, 32}) {
      GravityParams g2 = gp;
      g2.theta = 0.7;
      Octree tree;
      tree.build(ref.pos, ref.mass, 4);
      GroupLists L;
      build_group_lists(tree, g2, gs, L, pack);
      kern::GTParams tp = make_kernel_params(g2, n);
      tp.n_groups = static_cast<kern::uint>(L.groups.size());
      std::vector<kern::float4> cpu, gpu(n, kern::float4{NAN, NAN, NAN, NAN});
      evaluate_group_lists_cpu(L, tp, cpu);
      // Threadgroups wider than the largest group, as the host would size
      // them when rounding up, leave the extra threads idle.
      const unsigned threads = static_cast<unsigned>(L.max_group) + 3;
      mock::run_tree(L.bodies.data(), L.cells.data(), L.cell_list.data(), L.leaf_list.data(),
                     L.groups.data(), &tp, threads, gpu.data());
      const double d = max_rel(gpu, cpu);
      std::cout << "gt_tree, groups of " << gs << (pack ? " packed" : "")
                << ": max relative difference from the CPU evaluation " << d << "\n";
      CHECK(d == 0.0);  // same code, same order: bit identical
    }
  }
  return TEST_MAIN_RESULT();
}
