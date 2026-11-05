#include "galactic_tides/octree.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace galactic_tides {

namespace {
// Bounds the stack used by the walks (8 children per level).
constexpr int kMaxDepth = 60;

inline int octant(const Vec3 &p, const Vec3 &c) {
  return (p.x >= c.x ? 1 : 0) | (p.y >= c.y ? 2 : 0) | (p.z >= c.z ? 4 : 0);
}
}  // namespace

namespace {

using Node = Octree::Node;

// Everything the build needs besides the node array it writes into. index
// and scratch are shared by concurrent subtree builds, which touch disjoint
// ranges of them.
struct BuildContext {
  const std::vector<Vec3> &pos;
  const std::vector<double> &mass;
  std::vector<std::size_t> &index;
  std::vector<std::size_t> &scratch;
  std::uint32_t leaf_size;
};

bool stays_leaf(const Node &nd, std::uint32_t leaf_size, int depth) {
  const Vec3 &c = nd.center;
  return nd.end - nd.begin <= leaf_size || depth >= kMaxDepth ||
         nd.half < 1e-12 * (std::fabs(c.x) + std::fabs(c.y) + std::fabs(c.z) + 1e-300);
}

void leaf_moments(const BuildContext &ctx, Node &nd) {
  double m = 0.0;
  Vec3 com;
  for (std::uint32_t k = nd.begin; k < nd.end; ++k) {
    const std::size_t i = ctx.index[k];
    m += ctx.mass[i];
    com += ctx.mass[i] * ctx.pos[i];
  }
  com = m > 0 ? com / m : nd.center;
  double q[6] = {0, 0, 0, 0, 0, 0};
  for (std::uint32_t k = nd.begin; k < nd.end; ++k) {
    const std::size_t i = ctx.index[k];
    const double mi = ctx.mass[i];
    const Vec3 d = ctx.pos[i] - com;
    const double r2 = norm2(d);
    q[0] += mi * (3 * d.x * d.x - r2);
    q[1] += mi * (3 * d.y * d.y - r2);
    q[2] += mi * (3 * d.z * d.z - r2);
    q[3] += mi * 3 * d.x * d.y;
    q[4] += mi * 3 * d.x * d.z;
    q[5] += mi * 3 * d.y * d.z;
  }
  nd.mass = m;
  nd.com = com;
  std::copy(q, q + 6, nd.quad);
  nd.delta = norm(com - nd.center);
}

// Monopole and quadrupole of node ni from its children (parallel axis theorem).
void combine_children(std::vector<Node> &nodes, std::size_t ni) {
  Node &nd = nodes[ni];
  const std::size_t first = static_cast<std::size_t>(nd.first_child);
  const std::size_t last = first + static_cast<std::size_t>(nd.n_children);
  double m = 0.0;
  Vec3 com;
  for (std::size_t k = first; k < last; ++k) {
    m += nodes[k].mass;
    com += nodes[k].mass * nodes[k].com;
  }
  com = m > 0 ? com / m : nd.center;
  double q[6] = {0, 0, 0, 0, 0, 0};
  for (std::size_t k = first; k < last; ++k) {
    const Node &ch = nodes[k];
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

// Counting sort of node ni's bodies into octants, appending its (non-empty)
// children to nodes. Moments are not set.
void split(const BuildContext &ctx, std::vector<Node> &nodes, std::size_t ni) {
  const std::uint32_t begin = nodes[ni].begin, end = nodes[ni].end;
  const Vec3 center = nodes[ni].center;
  const double half = nodes[ni].half;
  std::array<std::uint32_t, 8> count{};
  for (std::uint32_t k = begin; k < end; ++k) ++count[octant(ctx.pos[ctx.index[k]], center)];
  std::array<std::uint32_t, 9> offset{};
  offset[0] = begin;
  for (int o = 0; o < 8; ++o) offset[o + 1] = offset[o] + count[o];
  std::array<std::uint32_t, 8> fill{};
  for (int o = 0; o < 8; ++o) fill[o] = offset[o];
  for (std::uint32_t k = begin; k < end; ++k) {
    const std::size_t i = ctx.index[k];
    ctx.scratch[fill[octant(ctx.pos[i], center)]++] = i;
  }
  std::copy(ctx.scratch.begin() + begin, ctx.scratch.begin() + end, ctx.index.begin() + begin);

  int nonempty = 0;
  for (int o = 0; o < 8; ++o) nonempty += count[o] > 0;
  const std::size_t first = nodes.size();
  nodes.resize(first + nonempty);  // invalidates references into nodes
  nodes[ni].first_child = static_cast<std::int32_t>(first);
  nodes[ni].n_children = nonempty;
  const double h2 = 0.5 * half;
  std::size_t c = first;
  for (int o = 0; o < 8; ++o) {
    if (!count[o]) continue;
    Node &ch = nodes[c++];
    ch.half = h2;
    ch.center = Vec3(center.x + ((o & 1) ? h2 : -h2), center.y + ((o & 2) ? h2 : -h2),
                     center.z + ((o & 4) ? h2 : -h2));
    ch.begin = offset[o];
    ch.end = offset[o + 1];
  }
}

// Builds the subtree under node ni (whose geometry and range are set);
// returns the deepest level reached.
int build_subtree(const BuildContext &ctx, std::vector<Node> &nodes, std::size_t ni, int depth) {
  if (stays_leaf(nodes[ni], ctx.leaf_size, depth)) {
    leaf_moments(ctx, nodes[ni]);
    return depth;
  }
  split(ctx, nodes, ni);
  const std::size_t first = static_cast<std::size_t>(nodes[ni].first_child);
  const int nch = nodes[ni].n_children;
  int deepest = depth;
  for (int c = 0; c < nch; ++c)
    deepest = std::max(deepest, build_subtree(ctx, nodes, first + c, depth + 1));
  combine_children(nodes, ni);
  return deepest;
}

int max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
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
  scratch_.resize(n);
  pos_.resize(n);
  mass_.resize(n);
  if (n == 0) return;

  double lx = pos[0].x, ly = pos[0].y, lz = pos[0].z, hx = lx, hy = ly, hz = lz;
#pragma omp parallel for schedule(static) reduction(min : lx, ly, lz) reduction(max : hx, hy, hz)
  for (long k = 0; k < static_cast<long>(n); ++k) {
    index_[k] = static_cast<std::size_t>(k);
    lx = std::min(lx, pos[k].x);
    ly = std::min(ly, pos[k].y);
    lz = std::min(lz, pos[k].z);
    hx = std::max(hx, pos[k].x);
    hy = std::max(hy, pos[k].y);
    hz = std::max(hz, pos[k].z);
  }
  const Vec3 lo(lx, ly, lz), hi(hx, hy, hz);
  double half = 0.0;
  for (int k = 0; k < 3; ++k) half = std::max(half, 0.5 * (hi[k] - lo[k]));
  half = half * (1.0 + 1e-9) + 1e-300;

  nodes_.reserve(2 * n / static_cast<std::size_t>(leaf_size_) + 16);
  nodes_.emplace_back();
  nodes_[0].center = 0.5 * (lo + hi);
  nodes_[0].half = half;
  nodes_[0].begin = 0;
  nodes_[0].end = static_cast<std::uint32_t>(n);
  const BuildContext ctx{pos, mass, index_, scratch_, static_cast<std::uint32_t>(leaf_size_)};

  if (n < parallel_threshold_) {
    depth_ = build_subtree(ctx, nodes_, 0, 0);
  } else {
    // Split the top levels serially until there are enough subtrees to share
    // out, build those in parallel into their own arrays, then splice them in.
    const std::size_t target = 8 * static_cast<std::size_t>(max_threads());
    std::vector<std::size_t> top;  // split serially, in breadth-first order
    std::vector<std::pair<std::size_t, int>> level{{0, 0}}, next;
    for (int d = 0; d < 4 && !level.empty() && level.size() < target; ++d) {
      next.clear();
      for (const auto &[ni, depth] : level) {
        if (stays_leaf(nodes_[ni], ctx.leaf_size, depth)) {
          leaf_moments(ctx, nodes_[ni]);
          depth_ = std::max(depth_, depth);
          continue;
        }
        split(ctx, nodes_, ni);
        top.push_back(ni);
        for (int c = 0; c < nodes_[ni].n_children; ++c)
          next.emplace_back(static_cast<std::size_t>(nodes_[ni].first_child + c), depth + 1);
      }
      level.swap(next);
    }

    std::vector<std::vector<Node>> subs(level.size());
    std::vector<int> deepest(level.size(), 0);
#pragma omp parallel for schedule(dynamic, 1)
    for (long k = 0; k < static_cast<long>(level.size()); ++k) {
      std::vector<Node> &sub = subs[k];
      const Node &root = nodes_[level[k].first];
      sub.reserve(2 * (root.end - root.begin) / ctx.leaf_size + 16);
      sub.push_back(root);
      deepest[k] = build_subtree(ctx, sub, 0, level[k].second);
    }

    // sub[0] replaces its frontier node; sub[1..] are appended, children
    // indices shifted to match.
    std::vector<std::size_t> base(level.size());
    std::size_t total = nodes_.size();
    for (std::size_t k = 0; k < level.size(); ++k) {
      base[k] = total;
      total += subs[k].size() - 1;
      depth_ = std::max(depth_, deepest[k]);
    }
    nodes_.resize(total);
#pragma omp parallel for schedule(dynamic, 1)
    for (long k = 0; k < static_cast<long>(level.size()); ++k) {
      const std::vector<Node> &sub = subs[k];
      auto place = [&](const Node &src) {
        Node nd = src;
        if (nd.first_child >= 0)
          nd.first_child = static_cast<std::int32_t>(base[k] + nd.first_child - 1);
        return nd;
      };
      nodes_[level[k].first] = place(sub[0]);
      for (std::size_t i = 1; i < sub.size(); ++i) nodes_[base[k] + i - 1] = place(sub[i]);
    }
    for (auto it = top.rbegin(); it != top.rend(); ++it) combine_children(nodes_, *it);
  }

#pragma omp parallel for schedule(static)
  for (long k = 0; k < static_cast<long>(n); ++k) {
    pos_[k] = pos[index_[k]];
    mass_[k] = mass[index_[k]];
  }
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
