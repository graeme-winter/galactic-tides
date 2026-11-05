// Times and checks the gravity solvers on a Hernquist galaxy.
//
//   galactic-tides-bench [n=100000] [theta=0.6] [eps=0.05] [quadrupole=1] [repeat=3]
//                   [sample=1000] [group_size=32] [group_pack=default]
//                   [backends=tree-cpu,group-cpu,group-metal,direct-metal]
//                   [simds=4] [gpu_trace=bench.gputrace]
//
// simds: groups per threadgroup for the Metal tree's SIMD kernel; 0 selects
// the threadgroup-tiled kernel instead.
//
// gpu_trace records one extra group-metal force evaluation into a .gputrace
// document for Xcode's GPU performance counters; run with
// MTL_CAPTURE_ENABLED=1 when not launched from Xcode.
//
// Accuracy is measured against double precision direct summation for a
// random sample of particles, so it stays affordable at large n.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "galactic_tides/gravity_backends.h"
#include "galactic_tides/hernquist_ics.h"
#include "galactic_tides/metal_backend.h"
#include "galactic_tides/params.h"

using namespace galactic_tides;

namespace {

std::vector<std::string> split(const std::string &s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty()) out.push_back(item);
  return out;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    Params params;
    auto rest = params.apply_overrides(std::vector<std::string>(argv + 1, argv + argc));
    if (!rest.empty()) {
      std::cerr << "usage: " << argv[0]
                << " [n=100000] [theta=0.6] [eps=0.05] [repeat=3] [sample=1000] [group_size=32]"
                   " [backends=tree-cpu,group-cpu,group-metal,direct-metal]\n";
      return 2;
    }
    const long n = params.get_int("n", 100000);
    const int repeat = static_cast<int>(params.get_int("repeat", 3));
    const long nsample = params.get_int("sample", 1000);
    const int group_size = static_cast<int>(params.get_int("group_size", 32));
    GravityParams gp;
    gp.theta = params.get_double("theta", 0.6);
    gp.eps = params.get_double("eps", 0.05);
    gp.quadrupole = params.get_bool("quadrupole", true);
    // -1: the device's default (packed on Metal, not on the CPU)
    const int pack = params.has("group_pack") ? (params.get_bool("group_pack", true) ? 1 : 0) : -1;
    const std::string gpu_trace = params.get_string("gpu_trace", "");
    const long simds = params.get_int("simds", 4);
    std::string def = "tree-cpu,group-cpu";
    if (metal::compiled_in()) def += ",group-metal,direct-metal";
    const auto backends = split(params.get_string("backends", def));
    for (const auto &k : params.unused()) std::fprintf(stderr, "warning: unused '%s'\n", k.c_str());

    // A galaxy with the default model's proportions.
    HernquistModel model;
    model.disk_n = static_cast<std::size_t>(n * 20000 / 85000);
    model.bulge_n = static_cast<std::size_t>(n * 5000 / 85000);
    model.halo_n = static_cast<std::size_t>(n) - model.disk_n - model.bulge_n;
    model.gravity = gp;
    std::printf("building a %ld particle galaxy...\n", n);
    const Particles base = make_hernquist_galaxy(model);

    // Reference for a sample of particles.
    std::mt19937_64 rng(1);
    std::vector<std::size_t> sample(base.size());
    for (std::size_t i = 0; i < sample.size(); ++i) sample[i] = i;
    std::shuffle(sample.begin(), sample.end(), rng);
    sample.resize(std::min<std::size_t>(sample.size(), static_cast<std::size_t>(nsample)));
    std::vector<Vec3> ref(sample.size());
    {
      const double eps2 = gp.eps * gp.eps;
#pragma omp parallel for schedule(dynamic)
      for (long s = 0; s < static_cast<long>(sample.size()); ++s) {
        const std::size_t i = sample[s];
        Vec3 a;
        for (std::size_t j = 0; j < base.size(); ++j) {
          if (j == i) continue;
          const Vec3 d = base.pos[i] - base.pos[j];
          const double inv = 1.0 / std::sqrt(norm2(d) + eps2);
          a -= (gp.G * base.mass[j] * inv * inv * inv) * d;
        }
        ref[s] = a;
      }
    }

    if (metal::compiled_in()) {
      try {
        metal::Context probe;
        std::printf("Metal device: %s\n", probe.device_name().c_str());
      } catch (const std::exception &e) {
        std::printf("Metal unavailable: %s\n", e.what());
      }
    }
    std::printf("theta %g, eps %g, quadrupole %s, %d timed repeats after a warm-up\n\n",
                gp.theta, gp.eps, gp.quadrupole ? "on" : "off", repeat);
    std::printf("%-14s %10s %12s %12s %12s   %s\n", "backend", "ms/force", "inter/part",
                "median err", "99% err", "breakdown (ms)");

    for (const auto &name : backends) {
      std::unique_ptr<ForceModel> model_ptr;
      GroupTreeGravity *group = nullptr;
      try {
        if (name == "tree-cpu") model_ptr = std::make_unique<TreeGravity>(gp);
        else if (name == "group-cpu" || name == "group-metal") {
          auto g = std::make_unique<GroupTreeGravity>(
              gp, name == "group-cpu" ? Device::CPU : Device::Metal, group_size, 8, pack);
          group = g.get();
          if (name == "group-metal") g->set_metal_simds(static_cast<unsigned>(simds));
          model_ptr = std::move(g);
        } else if (name == "direct-cpu") model_ptr = std::make_unique<DirectGravity>(gp, Device::CPU);
        else if (name == "direct-metal")
          model_ptr = std::make_unique<DirectGravity>(gp, Device::Metal);
        else throw std::runtime_error("unknown backend");
      } catch (const std::exception &e) {
        std::printf("%-14s skipped: %s\n", name.c_str(), e.what());
        continue;
      }
      Particles p = base;
      model_ptr->compute(p);  // warm-up (and Metal pipeline/buffer set-up)
      double best = 1e300;
      GroupTreeGravity::Timings tm;
      for (int r = 0; r < repeat; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        model_ptr->compute(p);
        const double t =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (t < best) {
          best = t;
          if (group) tm = group->timings();
        }
      }
      std::vector<double> err;
      for (std::size_t s = 0; s < sample.size(); ++s)
        err.push_back(norm(p.acc[sample[s]] - ref[s]) / norm(ref[s]));
      std::sort(err.begin(), err.end());
      const WalkStats st = model_ptr->stats();
      std::printf("%-14s %10.2f %12.0f %12.2e %12.2e", name.c_str(), best * 1e3,
                  double(st.cell_interactions + st.particle_interactions) / double(p.size()),
                  err[err.size() / 2], err[err.size() * 99 / 100]);
      if (group) {
        std::printf("   tree %.1f, lists %.1f, evaluate %.1f", tm.tree * 1e3, tm.lists * 1e3,
                    tm.evaluate * 1e3);
        if (name == "group-metal") {
          std::printf(" (gpu %.1f, %s kernel%s)", tm.gpu * 1e3, tm.kernel,
                      tm.zero_copy ? ", zero-copy" : ", copied");
          const auto &L = group->lists();
          std::printf("\n%-14s groups %zu, threads/threadgroup %d, mean fill %.1f, cells %.0f + "
                      "bodies %.0f per target, %.2e interactions/s on the GPU",
                      "", L.groups.size(), L.max_group, double(p.size()) / L.groups.size(),
                      double(L.cell_interactions) / p.size(), double(L.body_interactions) / p.size(),
                      double(L.cell_interactions + L.body_interactions) / std::max(tm.gpu, 1e-9));
        }
      }
      std::printf("\n");
      if (name == "group-metal" && !gpu_trace.empty()) {
        metal::start_gpu_capture(gpu_trace);
        model_ptr->compute(p);
        metal::stop_gpu_capture();
        std::printf("%-14s wrote %s (open in Xcode)\n", "", gpu_trace.c_str());
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
