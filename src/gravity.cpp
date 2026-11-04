#include "galactic_tides/gravity.h"

#include <cmath>

namespace galactic_tides {

WalkStats compute_gravity_tree(Particles &p, const GravityParams &gp, Octree &tree,
                               int leaf_size) {
  tree.build(p.pos, p.mass, leaf_size);
  const auto &order = tree.order();
  const long n = static_cast<long>(order.size());
  std::uint64_t ncell = 0, npart = 0;
  // Walk targets in tree order: neighbouring targets visit similar nodes.
#pragma omp parallel for schedule(dynamic, 256) reduction(+ : ncell, npart)
  for (long k = 0; k < n; ++k) {
    const std::size_t i = order[k];
    WalkStats s;
    tree.accel_pot(p.pos[i], i, gp, p.acc[i], p.pot[i], &s);
    ncell += s.cell_interactions;
    npart += s.particle_interactions;
  }
  WalkStats total;
  total.cell_interactions = ncell;
  total.particle_interactions = npart;
  return total;
}

void compute_gravity_direct(Particles &p, const GravityParams &gp) {
  const long n = static_cast<long>(p.size());
  const double eps2 = gp.eps * gp.eps;
#pragma omp parallel for schedule(static)
  for (long i = 0; i < n; ++i) {
    Vec3 a;
    double phi = 0.0;
    for (long j = 0; j < n; ++j) {
      if (j == i) continue;
      const Vec3 d = p.pos[i] - p.pos[j];
      const double inv = 1.0 / std::sqrt(norm2(d) + eps2);
      a -= (gp.G * p.mass[j] * inv * inv * inv) * d;
      phi -= gp.G * p.mass[j] * inv;
    }
    p.acc[i] = a;
    p.pot[i] = phi;
  }
}

}  // namespace galactic_tides
