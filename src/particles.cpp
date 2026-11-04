#include "galactic_tides/particles.h"

namespace galactic_tides {

const char *type_name(ParticleType t) {
  switch (t) {
    case ParticleType::Gas: return "gas";
    case ParticleType::Halo: return "halo";
    case ParticleType::Disk: return "disk";
    case ParticleType::Bulge: return "bulge";
    case ParticleType::Stars: return "stars";
    case ParticleType::Sink: return "sink";
  }
  return "unknown";
}

void Particles::resize(std::size_t n) {
  pos.resize(n);
  vel.resize(n);
  acc.resize(n);
  pot.resize(n);
  mass.resize(n);
  id.resize(n);
  type.resize(n);
}

void Particles::reserve(std::size_t n) {
  pos.reserve(n);
  vel.reserve(n);
  acc.reserve(n);
  pot.reserve(n);
  mass.reserve(n);
  id.reserve(n);
  type.reserve(n);
}

void Particles::push_back(const Vec3 &p, const Vec3 &v, double m, std::uint64_t ident,
                          ParticleType t) {
  pos.push_back(p);
  vel.push_back(v);
  acc.emplace_back();
  pot.push_back(0.0);
  mass.push_back(m);
  id.push_back(ident);
  type.push_back(t);
}

std::array<std::size_t, kNumParticleTypes> Particles::count_by_type() const {
  std::array<std::size_t, kNumParticleTypes> n{};
  for (auto t : type) ++n[static_cast<int>(t)];
  return n;
}

}  // namespace galactic_tides
