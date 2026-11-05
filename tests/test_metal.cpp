// Runs the Metal kernels on the GPU and compares them with the CPU: the
// emulated kernels (same arithmetic, so agreement should be at single
// precision rounding) and the double precision reference. Built only with
// Metal; reports and passes if the machine has no usable GPU.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

#include "check.h"
#include "galactic_tides/gravity.h"
#include "galactic_tides/gravity_backends.h"
#include "galactic_tides/metal_backend.h"

using namespace galactic_tides;

namespace {

Particles make_test_system(std::size_t n, unsigned seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  Particles p;
  for (std::size_t i = 0; i < n; ++i) {
    Vec3 centre;
    double a = 1.0;
    if (i % 10 == 0) {
      centre = (i % 20 == 0) ? Vec3(4, 0, 0) : Vec3(-1, 3, 2);
      a = 0.1;
    }
    double x = std::sqrt(u(rng) * 0.95);
    double r = a * x / (1 - x);
    double cz = 2 * u(rng) - 1, ph = 2 * M_PI * u(rng), s = std::sqrt(1 - cz * cz);
    p.push_back(centre + r * Vec3(s * std::cos(ph), s * std::sin(ph), cz), Vec3(), 1.0 / n, i,
                ParticleType::Halo);
  }
  return p;
}

struct Errors {
  double median, max, pot_max;
};

Errors compare(const Particles &a, const Particles &b) {
  std::vector<double> e, ep;
  for (std::size_t i = 0; i < a.size(); ++i) {
    e.push_back(norm(a.acc[i] - b.acc[i]) / norm(b.acc[i]));
    ep.push_back(std::fabs(a.pot[i] / b.pot[i] - 1));
  }
  std::sort(e.begin(), e.end());
  return {e[e.size() / 2], e.back(), *std::max_element(ep.begin(), ep.end())};
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main() {
  try {
    metal::Context probe;
    std::cout << "Metal device: " << probe.device_name() << "\n";
  } catch (const std::exception &e) {
    std::cout << "skipping: " << e.what() << "\n";
    return 0;
  }

  // Odd size, so the last threadgroup is partly empty.
  const std::size_t n = 5003;
  Particles ref = make_test_system(n, 5);
  GravityParams gp;
  gp.eps = 0.01;
  compute_gravity_direct(ref, gp);

  // ---- brute force -----------------------------------------------------------
  {
    Particles p = ref;
    DirectGravity gpu(gp, Device::Metal);
    gpu.compute(p);  // includes pipeline warm-up
    auto t0 = std::chrono::steady_clock::now();
    gpu.compute(p);
    const double t = seconds_since(t0);
    const Errors e = compare(p, ref);
    std::cout << "metal direct vs double: median " << e.median << " max " << e.max
              << " potential max " << e.pot_max << "  (" << t * 1e3 << " ms)\n";
    CHECK(e.median < 1e-5);
    CHECK(e.max < 1e-3);
    CHECK(e.pot_max < 1e-5);
  }
  // ---- tree -----------------------------------------------------------------
  // Same lists on both sides (same packing), so the GPU should reproduce the
  // CPU evaluation of the shared kernel to single precision rounding.
  for (int group_size : {32, 64}) {
    for (double theta : {0.5, 0.8}) {
      GravityParams g2 = gp;
      g2.theta = theta;
      Particles cpu = ref, gpu = ref;
      GroupTreeGravity on_cpu(g2, Device::CPU, group_size, 8, 1);
      GroupTreeGravity on_gpu(g2, Device::Metal, group_size, 8, 1);
      on_cpu.compute(cpu);
      on_gpu.compute(gpu);
      auto t0 = std::chrono::steady_clock::now();
      on_gpu.compute(gpu);
      const double t = seconds_since(t0);
      const Errors same = compare(gpu, cpu), acc = compare(gpu, ref);
      const auto &tm = on_gpu.timings();
      std::cout << "metal tree, groups of " << group_size << ", theta " << theta
                << ": vs cpu kernel median " << same.median << " max " << same.max
                << " | vs direct median " << acc.median << " max " << acc.max << "  (" << t * 1e3
                << " ms: tree " << tm.tree * 1e3 << ", lists " << tm.lists * 1e3 << ", gpu "
                << tm.evaluate * 1e3 << " of which gpu " << tm.gpu * 1e3
                << (tm.zero_copy ? ", zero-copy" : ", copied") << ")\n";
      CHECK(same.median < 1e-5);
      CHECK(same.max < 1e-3);
      CHECK(acc.median < 3e-3 * theta * theta);
    }
  }
  // The widest groups the tree kernel accepts, and one beyond: refused.
  {
    GravityParams g2 = gp;
    Particles cpu = ref, gpu = ref;
    GroupTreeGravity on_cpu(g2, Device::CPU, static_cast<int>(GT_TREE_TILE), 8, 1);
    GroupTreeGravity on_gpu(g2, Device::Metal, static_cast<int>(GT_TREE_TILE), 8, 1);
    on_cpu.compute(cpu);
    on_gpu.compute(gpu);
    CHECK(compare(gpu, cpu).max < 1e-3);
    bool threw = false;
    try {
      GroupTreeGravity too_wide(g2, Device::Metal, static_cast<int>(GT_TREE_TILE) + 1);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    CHECK(threw);
  }
  return TEST_MAIN_RESULT();
}
