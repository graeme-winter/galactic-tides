#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "galactic_tides/vec3.h"

namespace galactic_tides {

// Particle types follow the GADGET convention so that snapshots are readable
// by the usual tools (pynbody, yt, ...) and so that ICs from other generators
// (DICE, MakeNewDisk, AGAMA via conversion) slot straight in.
enum class ParticleType : std::uint8_t {
  Gas = 0,    // reserved for SPH
  Halo = 1,
  Disk = 2,
  Bulge = 3,
  Stars = 4,  // reserved (star formation)
  Sink = 5,   // reserved (black holes / sinks / collision products)
};
constexpr int kNumParticleTypes = 6;

const char *type_name(ParticleType t);

// Structure of arrays. Collisionless fields only for now; hydro fields
// (internal energy, density, smoothing length) will live alongside these and
// only be meaningful for ParticleType::Gas.
struct Particles {
  std::vector<Vec3> pos;
  std::vector<Vec3> vel;
  std::vector<Vec3> acc;
  std::vector<double> pot;  // gravitational potential per unit mass
  std::vector<double> mass;
  std::vector<std::uint64_t> id;
  std::vector<ParticleType> type;

  std::size_t size() const { return pos.size(); }
  void resize(std::size_t n);
  void reserve(std::size_t n);
  void push_back(const Vec3 &p, const Vec3 &v, double m, std::uint64_t ident, ParticleType t);

  std::array<std::size_t, kNumParticleTypes> count_by_type() const;
};

}  // namespace galactic_tides
