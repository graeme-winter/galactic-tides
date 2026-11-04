// Evolve a snapshot with the tree code.
//
//   galactic-tides-run params.par [key=value ...]
//
// Any snapshot can be used as the starting point, so restarting is just
// ics=output/snap_0042.hdf5; numbering continues from the snapshot's time.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "galactic_tides/integrator.h"
#include "galactic_tides/params.h"
#include "galactic_tides/snapshot_io.h"

using namespace galactic_tides;

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      std::cerr << "usage: " << argv[0] << " params.par [key=value ...]\n";
      return 2;
    }
    Params params;
    params.load_file(argv[1]);
    auto rest = params.apply_overrides(std::vector<std::string>(argv + 2, argv + argc));
    if (!rest.empty()) throw std::runtime_error("unexpected argument: " + rest[0]);

    const std::string ics = params.require_string("ics");
    const std::string outdir = params.get_string("output_dir", "output");
    const std::string prefix = params.get_string("snapshot_prefix", "snap");
    const double dt = params.get_double("dt", 0.01);
    const double t_end = params.get_double("t_end", 10.0);
    const double snap_every = params.get_double("snapshot_interval", 1.0);
    const long log_every = params.get_int("log_interval", 10);
    const int leaf_size = static_cast<int>(params.get_int("leaf_size", 8));
    GravityParams gp;
    gp.G = params.get_double("G", gp.G);
    gp.eps = params.get_double("eps", gp.eps);
    gp.theta = params.get_double("theta", gp.theta);
    gp.quadrupole = params.get_bool("quadrupole", gp.quadrupole);
    if (dt <= 0 || snap_every <= 0) throw std::runtime_error("dt and snapshot_interval must be > 0");
    for (const auto &k : params.unused())
      std::fprintf(stderr, "warning: parameter '%s' was not used\n", k.c_str());

    double t = 0.0;
    Particles p = read_snapshot(ics, &t);
    std::filesystem::create_directories(outdir);
    const auto counts = p.count_by_type();
    std::printf("read %zu particles from %s (t = %g):", p.size(), ics.c_str(), t);
    for (int k = 0; k < kNumParticleTypes; ++k)
      if (counts[k]) std::printf(" %s %zu", type_name(static_cast<ParticleType>(k)), counts[k]);
    std::printf("\ntheta %g  eps %g  quadrupole %s  dt %g  t_end %g\n", gp.theta, gp.eps,
                gp.quadrupole ? "yes" : "no", dt, t_end);

    TreeGravity gravity(gp, leaf_size);
    LeapfrogKDK leapfrog(gravity);
    leapfrog.prime(p);
    const Diagnostics d0 = compute_diagnostics(p, t);

    // Number snapshots by time so that a run restarted from a snapshot
    // carries on the sequence rather than overwriting it.
    int snap = static_cast<int>(
        params.get_int("snapshot_start", std::lround(t / snap_every)));
    auto write = [&]() {
      char name[64];
      std::snprintf(name, sizeof name, "%s_%04d.hdf5", prefix.c_str(), snap++);
      write_snapshot((std::filesystem::path(outdir) / name).string(), p, t, params.used());
    };
    double next_snap = t;
    if (t >= next_snap - 1e-12 * dt) {
      write();
      next_snap += snap_every;
    }

    std::printf("%8s %10s %14s %11s %8s %11s %11s %9s %8s\n", "step", "time", "energy", "dE/E0",
                "2K/|W|", "|P|", "Lz", "int/part", "s/step");
    auto log = [&](long step, double secs) {
      const Diagnostics d = compute_diagnostics(p, t);
      const auto &s = gravity.last_stats();
      std::printf("%8ld %10.4f %14.7e %11.3e %8.4f %11.3e %11.4e %9.0f %8.3f\n", step, t,
                  d.total(), d.total() / d0.total() - 1, d.virial_ratio(), norm(d.momentum),
                  d.angular_momentum.z,
                  double(s.cell_interactions + s.particle_interactions) / double(p.size()), secs);
      std::fflush(stdout);
    };
    log(0, 0.0);

    const long nsteps = static_cast<long>(std::ceil((t_end - t) / dt - 1e-9));
    auto clock_start = std::chrono::steady_clock::now();
    for (long step = 1; step <= nsteps; ++step) {
      leapfrog.step(p, dt);
      t += dt;
      if (t >= next_snap - 1e-9 * dt) {
        write();
        next_snap += snap_every;
      }
      if (step % log_every == 0 || step == nsteps) {
        auto now = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(now - clock_start).count();
        log(step, secs / double(step % log_every == 0 ? log_every : (step % log_every)));
        clock_start = now;
      }
    }
    std::printf("done: last snapshot %s_%04d.hdf5 in %s\n", prefix.c_str(), snap - 1,
                outdir.c_str());
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
