#include "galactic_tides/integrator.h"

namespace galactic_tides {

void LeapfrogKDK::step(Particles &p, double dt) {
  const long n = static_cast<long>(p.size());
  const double h = 0.5 * dt;
#pragma omp parallel for schedule(static)
  for (long i = 0; i < n; ++i) {
    p.vel[i] += h * p.acc[i];
    p.pos[i] += dt * p.vel[i];
  }
  forces_.compute(p);
#pragma omp parallel for schedule(static)
  for (long i = 0; i < n; ++i) p.vel[i] += h * p.acc[i];
}

Diagnostics compute_diagnostics(const Particles &p, double time) {
  Diagnostics d;
  d.time = time;
  double mtot = 0.0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    const double m = p.mass[i];
    d.kinetic += 0.5 * m * norm2(p.vel[i]);
    d.potential += 0.5 * m * p.pot[i];
    d.momentum += m * p.vel[i];
    d.angular_momentum += m * cross(p.pos[i], p.vel[i]);
    d.centre_of_mass += m * p.pos[i];
    mtot += m;
  }
  if (mtot > 0) d.centre_of_mass /= mtot;
  return d;
}

}  // namespace galactic_tides
