#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "galactic_tides/vec3.h"

namespace galactic_tides {

struct GravityParams {
  double G = 1.0;
  double theta = 0.6;      // opening angle
  double eps = 0.05;       // Plummer softening length
  bool quadrupole = true;  // include quadrupole moments of accepted cells
};

// Counts of interactions from a force evaluation: a cheap way to see what the
// opening angle is costing you.
struct WalkStats {
  std::uint64_t cell_interactions = 0;
  std::uint64_t particle_interactions = 0;
  WalkStats &operator+=(const WalkStats &o) {
    cell_interactions += o.cell_interactions;
    particle_interactions += o.particle_interactions;
    return *this;
  }
};

// Barnes & Hut (1986) octree with bucket leaves.
//
// Nodes own a contiguous range [begin, end) of a tree-ordered copy of the
// particle positions and masses, so leaves are cache friendly and the
// tree-ordering can be reused for walking particles in spatial order. The
// children of a node are stored contiguously.
//
// Cells carry monopole and (traceless) quadrupole moments about their centre
// of mass. The opening criterion is Barnes (1994): a cell of side l whose
// centre of mass is offset by delta from its geometric centre is accepted
// when d > l / theta + delta, and is always opened if the target lies inside
// it. This avoids the classic failure of the plain l/d < theta test for
// cells with lopsided mass distributions.
//
// The same tree answers fixed-radius neighbour queries (for_each_within),
// which is what SPH and collision detection will need.
class Octree {
 public:
  static constexpr std::size_t kNoSelf = std::numeric_limits<std::size_t>::max();

  struct Node {
    Vec3 center;  // geometric centre of the cube
    double half = 0.0;  // half side length
    Vec3 com;
    double mass = 0.0;
    double quad[6] = {0, 0, 0, 0, 0, 0};  // xx, yy, zz, xy, xz, yz
    double delta = 0.0;  // |com - center|
    std::int32_t first_child = -1;
    std::int32_t n_children = 0;
    std::uint32_t begin = 0, end = 0;  // range in tree order
    bool is_leaf() const { return n_children == 0; }
  };

  void build(const std::vector<Vec3> &pos, const std::vector<double> &mass, int leaf_size = 8);

  // Acceleration and potential at x from all particles, skipping particle
  // `self` (an original index), or nothing if self == kNoSelf.
  void accel_pot(const Vec3 &x, std::size_t self, const GravityParams &gp, Vec3 &acc,
                 double &pot, WalkStats *stats = nullptr) const;

  // Calls f(index, position, distance_squared) for every particle within
  // radius r of c (inclusive). Index is the original particle index.
  template <class F>
  void for_each_within(const Vec3 &c, double r, F &&f) const;

  const std::vector<Node> &nodes() const { return nodes_; }
  // tree order -> original index
  const std::vector<std::size_t> &order() const { return index_; }
  // positions and masses in tree order
  const std::vector<Vec3> &tree_positions() const { return pos_; }
  const std::vector<double> &tree_masses() const { return mass_; }
  std::size_t size() const { return index_.size(); }
  int depth() const { return depth_; }

  // Trees for at least this many particles (default 20000) are built in
  // parallel: the top levels serially, then subtrees concurrently. The
  // result is the same tree, with nodes in a different order.
  void set_parallel_threshold(std::size_t n) { parallel_threshold_ = n; }

 private:

  std::vector<Node> nodes_;
  std::vector<std::size_t> index_;
  std::vector<std::size_t> scratch_;
  std::vector<Vec3> pos_;  // tree order
  std::vector<double> mass_;
  int leaf_size_ = 8;
  int depth_ = 0;
  std::size_t parallel_threshold_ = 20000;
};

template <class F>
void Octree::for_each_within(const Vec3 &c, double r, F &&f) const {
  if (nodes_.empty()) return;
  const double r2 = r * r;
  std::size_t stack[1024];
  int sp = 0;
  stack[sp++] = 0;
  while (sp > 0) {
    const Node &n = nodes_[stack[--sp]];
    double d2 = 0.0;
    for (int k = 0; k < 3; ++k) {
      double d = std::fabs(c[k] - n.center[k]) - n.half;
      if (d > 0) d2 += d * d;
    }
    if (d2 > r2) continue;
    if (n.is_leaf()) {
      for (std::uint32_t k = n.begin; k < n.end; ++k) {
        double dd = norm2(pos_[k] - c);
        if (dd <= r2) f(index_[k], pos_[k], dd);
      }
    } else {
      for (int k = 0; k < n.n_children; ++k) stack[sp++] = n.first_child + k;
    }
  }
}

}  // namespace galactic_tides
