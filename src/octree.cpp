#include "galactic_tides/octree.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace galactic_tides {

namespace {
// Bounds the stack used by the walks (8 children per level).
constexpr int kMaxDepth = 60;

inline int octant(const Vec3 &p, const Vec3 &c) {
  return (p.x >= c.x ? 1 : 0) | (p.y >= c.y ? 2 : 0) | (p.z >= c.z ? 4 : 0);
}
}  // namespace

void Octree::build(const std::vector<Vec3> &pos, const std::vector<double> &mass,
                   int leaf_size) {
  if (pos.size() != mass.size()) throw std::invalid_argument("Octree::build: size mismatch");
  if (pos.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("Octree::build: too many particles");
  leaf_size_ = std::max(1, leaf_size);
  depth_ = 0;
  nodes_.clear();
  const std::size_t n = pos.size();
  index_.resize(n);
  std::iota(index_.begin(), index_.end(), std::size_t{0});
  scratch_.resize(n);
  pos_.resize(n);
  mass_.resize(n);
  if (n == 0) return;

  Vec3 lo = pos[0], hi = pos[0];
  for (const auto &p : pos) {
    for (int k = 0; k < 3; ++k) {
      lo[k] = std::min(lo[k], p[k]);
      hi[k] = std::max(hi[k], p[k]);
    }
  }
  double half = 0.0;
  for (int k = 0; k < 3; ++k) half = std::max(half, 0.5 * (hi[k] - lo[k]));
  half = half * (1.0 + 1e-9) + 1e-300;

  nodes_.reserve(2 * n / static_cast<std::size_t>(leaf_size_) + 16);
  nodes_.emplace_back();
  nodes_[0].center = 0.5 * (lo + hi);
  nodes_[0].half = half;
  nodes_[0].begin = 0;
  nodes_[0].end = static_cast<std::uint32_t>(n);
  build_into(0, pos, mass, 0);

  for (std::size_t k = 0; k < n; ++k) {
    pos_[k] = pos[index_[k]];
    mass_[k] = mass[index_[k]];
  }
  scratch_.clear();
  scratch_.shrink_to_fit();
}

void Octree::build_into(std::size_t ni, const std::vector<Vec3> &pos,
                        const std::vector<double> &mass, int depth) {
  depth_ = std::max(depth_, depth);
  const std::uint32_t begin = nodes_[ni].begin, end = nodes_[ni].end;

  auto set_moments_from_particles = [&](Node &nd) {
    double m = 0.0;
    Vec3 com;
    for (std::uint32_t k = nd.begin; k < nd.end; ++k) {
      const std::size_t i = index_[k];
      m += mass[i];
      com += mass[i] * pos[i];
    }
    com = m > 0 ? com / m : nd.center;
    double q[6] = {0, 0, 0, 0, 0, 0};
    for (std::uint32_t k = nd.begin; k < nd.end; ++k) {
      const std::size_t i = index_[k];
      const Vec3 d = pos[i] - com;
      const double r2 = norm2(d);
      q[0] += mass[i] * (3 * d.x * d.x - r2);
      q[1] += mass[i] * (3 * d.y * d.y - r2);
      q[2] += mass[i] * (3 * d.z * d.z - r2);
      q[3] += mass[i] * 3 * d.x * d.y;
      q[4] += mass[i] * 3 * d.x * d.z;
      q[5] += mass[i] * 3 * d.y * d.z;
    }
    nd.mass = m;
    nd.com = com;
    std::copy(q, q + 6, nd.quad);
    nd.delta = norm(com - nd.center);
  };

  const Vec3 center = nodes_[ni].center;
  const double half = nodes_[ni].half;
  if (end - begin <= static_cast<std::uint32_t>(leaf_size_) || depth >= kMaxDepth ||
      half < 1e-12 * (std::fabs(center.x) + std::fabs(center.y) + std::fabs(center.z) + 1e-300)) {
    set_moments_from_particles(nodes_[ni]);
    return;
  }

  // Counting sort of [begin, end) into octants.
  std::array<std::uint32_t, 8> count{};
  for (std::uint32_t k = begin; k < end; ++k) ++count[octant(pos[index_[k]], center)];
  std::array<std::uint32_t, 9> offset{};
  offset[0] = begin;
  for (int o = 0; o < 8; ++o) offset[o + 1] = offset[o] + count[o];
  std::array<std::uint32_t, 8> fill{};
  for (int o = 0; o < 8; ++o) fill[o] = offset[o];
  for (std::uint32_t k = begin; k < end; ++k) {
    const std::size_t i = index_[k];
    scratch_[fill[octant(pos[i], center)]++] = i;
  }
  std::copy(scratch_.begin() + begin, scratch_.begin() + end, index_.begin() + begin);

  int nonempty = 0;
  for (int o = 0; o < 8; ++o) nonempty += count[o] > 0;
  const std::size_t first = nodes_.size();
  nodes_.resize(first + nonempty);  // invalidates references into nodes_
  nodes_[ni].first_child = static_cast<std::int32_t>(first);
  nodes_[ni].n_children = nonempty;

  const double h2 = 0.5 * half;
  std::size_t c = first;
  for (int o = 0; o < 8; ++o) {
    if (!count[o]) continue;
    Node &ch = nodes_[c];
    ch.half = h2;
    ch.center = Vec3(center.x + ((o & 1) ? h2 : -h2), center.y + ((o & 2) ? h2 : -h2),
                     center.z + ((o & 4) ? h2 : -h2));
    ch.begin = offset[o];
    ch.end = offset[o + 1];
    ++c;
  }
  for (std::size_t k = first; k < first + nonempty; ++k) build_into(k, pos, mass, depth + 1);

  // Combine children: monopole, then quadrupole via the parallel axis theorem.
  Node &nd = nodes_[ni];
  double m = 0.0;
  Vec3 com;
  for (std::size_t k = first; k < first + nonempty; ++k) {
    m += nodes_[k].mass;
    com += nodes_[k].mass * nodes_[k].com;
  }
  com = m > 0 ? com / m : nd.center;
  double q[6] = {0, 0, 0, 0, 0, 0};
  for (std::size_t k = first; k < first + nonempty; ++k) {
    const Node &ch = nodes_[k];
    const Vec3 d = ch.com - com;
    const double r2 = norm2(d);
    q[0] += ch.quad[0] + ch.mass * (3 * d.x * d.x - r2);
    q[1] += ch.quad[1] + ch.mass * (3 * d.y * d.y - r2);
    q[2] += ch.quad[2] + ch.mass * (3 * d.z * d.z - r2);
    q[3] += ch.quad[3] + ch.mass * 3 * d.x * d.y;
    q[4] += ch.quad[4] + ch.mass * 3 * d.x * d.z;
    q[5] += ch.quad[5] + ch.mass * 3 * d.y * d.z;
  }
  nd.mass = m;
  nd.com = com;
  std::copy(q, q + 6, nd.quad);
  nd.delta = norm(com - nd.center);
}

void Octree::accel_pot(const Vec3 &x, std::size_t self, const GravityParams &gp, Vec3 &acc,
                       double &pot, WalkStats *stats) const {
  acc = Vec3();
  pot = 0.0;
  if (nodes_.empty()) return;
  const double eps2 = gp.eps * gp.eps;
  const double inv_theta = 1.0 / gp.theta;
  std::uint64_t ncell = 0, npart = 0;

  std::size_t stack[1024];
  int sp = 0;
  stack[sp++] = 0;
  double ax = 0, ay = 0, az = 0, phi = 0;
  while (sp > 0) {
    const Node &n = nodes_[stack[--sp]];
    const Vec3 dx = x - n.com;
    const double r2 = norm2(dx);
    const double crit = 2.0 * n.half * inv_theta + n.delta;
    const double slack = n.half * 1.0001;
    const bool inside = std::fabs(x.x - n.center.x) <= slack &&
                        std::fabs(x.y - n.center.y) <= slack &&
                        std::fabs(x.z - n.center.z) <= slack;
    if (!inside && r2 > crit * crit) {
      // Accept the cell as a multipole.
      const double r2s = r2 + eps2;
      const double inv = 1.0 / std::sqrt(r2s);
      const double inv3 = inv * inv * inv;
      const double gm3 = gp.G * n.mass * inv3;
      ax -= gm3 * dx.x;
      ay -= gm3 * dx.y;
      az -= gm3 * dx.z;
      phi -= gp.G * n.mass * inv;
      if (gp.quadrupole) {
        const double *q = n.quad;
        const double qx = q[0] * dx.x + q[3] * dx.y + q[4] * dx.z;
        const double qy = q[3] * dx.x + q[1] * dx.y + q[5] * dx.z;
        const double qz = q[4] * dx.x + q[5] * dx.y + q[2] * dx.z;
        const double rqr = dx.x * qx + dx.y * qy + dx.z * qz;
        const double inv5 = inv3 * inv * inv;
        const double inv7 = inv5 * inv * inv;
        ax += gp.G * (qx * inv5 - 2.5 * rqr * dx.x * inv7);
        ay += gp.G * (qy * inv5 - 2.5 * rqr * dx.y * inv7);
        az += gp.G * (qz * inv5 - 2.5 * rqr * dx.z * inv7);
        phi -= 0.5 * gp.G * rqr * inv5;
      }
      ++ncell;
    } else if (n.is_leaf()) {
      for (std::uint32_t k = n.begin; k < n.end; ++k) {
        if (index_[k] == self) continue;
        const Vec3 d = x - pos_[k];
        const double r2s = norm2(d) + eps2;
        const double inv = 1.0 / std::sqrt(r2s);
        const double gm3 = gp.G * mass_[k] * inv * inv * inv;
        ax -= gm3 * d.x;
        ay -= gm3 * d.y;
        az -= gm3 * d.z;
        phi -= gp.G * mass_[k] * inv;
        ++npart;
      }
    } else {
      for (int k = 0; k < n.n_children; ++k) stack[sp++] = n.first_child + k;
    }
  }
  acc = Vec3(ax, ay, az);
  pot = phi;
  if (stats) {
    stats->cell_interactions += ncell;
    stats->particle_interactions += npart;
  }
}

}  // namespace galactic_tides
