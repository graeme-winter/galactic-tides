#pragma once

#include "galactic_tides/octree.h"
#include "galactic_tides/particles.h"

namespace galactic_tides {

// Builds `tree` from the particles and fills p.acc and p.pot.
WalkStats compute_gravity_tree(Particles &p, const GravityParams &gp, Octree &tree,
                               int leaf_size = 8);

// O(N^2) reference with the same softening. For tests and small N.
void compute_gravity_direct(Particles &p, const GravityParams &gp);

}  // namespace galactic_tides
