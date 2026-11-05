#pragma once

#include "galactic_tides/gravity.h"
#include "galactic_tides/octree.h"
#include "galactic_tides/particles.h"

namespace galactic_tides {

// Something that fills p.acc (and p.pot). Gravity is the only one for now;
// SPH pressure forces and collision handling will be further implementations
// composed alongside it.
class ForceModel {
 public:
  virtual ~ForceModel() = default;
  virtual void compute(Particles &p) = 0;
  // Interaction counts from the last compute(), for logging.
  virtual WalkStats stats() const { return {}; }
};

class TreeGravity : public ForceModel {
 public:
  TreeGravity(const GravityParams &gp, int leaf_size = 8) : gp_(gp), leaf_size_(leaf_size) {}
  void compute(Particles &p) override { stats_ = compute_gravity_tree(p, gp_, tree_, leaf_size_); }
  WalkStats stats() const override { return stats_; }
  const WalkStats &last_stats() const { return stats_; }
  const Octree &tree() const { return tree_; }
  const GravityParams &params() const { return gp_; }

 private:
  GravityParams gp_;
  int leaf_size_;
  Octree tree_;
  WalkStats stats_;
};

// Kick-drift-kick leapfrog with a single global timestep: symplectic,
// time-reversible, second order. Forces must be current on entry to step(),
// i.e. call prime() once first (or after modifying particles by hand).
class LeapfrogKDK {
 public:
  explicit LeapfrogKDK(ForceModel &forces) : forces_(forces) {}
  void prime(Particles &p) { forces_.compute(p); }
  void step(Particles &p, double dt);

 private:
  ForceModel &forces_;
};

struct Diagnostics {
  double time = 0.0;
  double kinetic = 0.0;
  double potential = 0.0;  // 0.5 * sum m phi
  double total() const { return kinetic + potential; }
  double virial_ratio() const { return potential != 0.0 ? -2.0 * kinetic / potential : 0.0; }
  Vec3 momentum;
  Vec3 angular_momentum;
  Vec3 centre_of_mass;
};

// Uses p.pot, so call after a force evaluation.
Diagnostics compute_diagnostics(const Particles &p, double time);

}  // namespace galactic_tides
