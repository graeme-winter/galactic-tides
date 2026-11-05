#include "galactic_tides/group_lists.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace galactic_tides {

namespace {

struct GroupSpec {
  std::uint32_t begin, end;  // targets, tree order
};

struct GroupWork {
  std::vector<kern::uint> cells;
  std::vector<kern::uint> bodies;
  void add_bodies(std::uint32_t begin, std::uint32_t end) {
    for (std::uint32_t j = begin; j < end; ++j) bodies.push_back(j);
  }
};

}  // namespace

void build_group_lists(const Octree &tree, const GravityParams &gp, int group_size,
                       GroupLists &out, bool pack) {
  if (group_size < 1) throw std::invalid_argument("group_size must be at least 1");
  const auto &nodes = tree.nodes();
  const auto &pos = tree.tree_positions();
  const auto &mass = tree.tree_masses();
  const std::size_t n = pos.size();
  if (n > 0xffffffffu || nodes.size() > 0xffffffffu)
    throw std::runtime_error("too many particles for the 32 bit GPU indices");

  out.groups.clear();
  out.cell_list.clear();
  out.body_list.clear();
  out.max_group = 0;
  out.cell_interactions = out.body_interactions = 0;
  out.bodies.resize(n);
  out.cells.resize(nodes.size());
  if (n == 0) return;

  // Bodies and cells relative to the root centre, in single precision.
  out.origin = nodes[0].center;
  for (std::size_t k = 0; k < n; ++k) {
    const Vec3 d = pos[k] - out.origin;
    out.bodies[k] = kern::float4{float(d.x), float(d.y), float(d.z), float(mass[k])};
  }
  for (std::size_t k = 0; k < nodes.size(); ++k) {
    const auto &nd = nodes[k];
    const Vec3 c = nd.com - out.origin;
    const double *q = nd.quad;
    out.cells[k].com_mass = kern::float4{float(c.x), float(c.y), float(c.z), float(nd.mass)};
    out.cells[k].q0 = kern::float4{float(q[0]), float(q[1]), float(q[2]), float(q[3])};
    out.cells[k].q1 = kern::float4{float(q[4]), float(q[5]), 0.0f, 0.0f};
  }

  // Choose groups: the highest nodes holding at most group_size bodies. A
  // leaf can hold more (coincident particles at maximum depth, or a leaf
  // size above the group size); such leaves are cut into chunks.
  std::vector<GroupSpec> specs;
  {
    std::vector<std::size_t> stack{0};
    while (!stack.empty()) {
      const auto &nd = nodes[stack.back()];
      stack.pop_back();
      const std::uint32_t count = nd.end - nd.begin;
      if (count <= static_cast<std::uint32_t>(group_size) || nd.is_leaf()) {
        for (std::uint32_t b = nd.begin; b < nd.end; b += group_size)
          specs.push_back({b, std::min<std::uint32_t>(nd.end, b + group_size)});
      } else {
        // Reverse, so groups come out in tree order.
        for (int c = nd.n_children - 1; c >= 0; --c) stack.push_back(nd.first_child + c);
      }
    }
  }

  if (pack && !specs.empty()) {
    std::vector<GroupSpec> packed{specs[0]};
    for (std::size_t k = 1; k < specs.size(); ++k) {
      GroupSpec &cur = packed.back();
      if (specs[k].end - cur.begin <= static_cast<std::uint32_t>(group_size))
        cur.end = specs[k].end;  // specs tile [0, n) in order, so this stays contiguous
      else
        packed.push_back(specs[k]);
    }
    specs.swap(packed);
  }

  // Walk once per group.
  const double inv_theta = 1.0 / gp.theta;
  std::vector<GroupWork> work(specs.size());
#pragma omp parallel for schedule(dynamic, 16)
  for (long g = 0; g < static_cast<long>(specs.size()); ++g) {
    const GroupSpec &spec = specs[g];
    Vec3 lo = pos[spec.begin], hi = lo;
    for (std::uint32_t k = spec.begin; k < spec.end; ++k)
      for (int a = 0; a < 3; ++a) {
        lo[a] = std::min(lo[a], pos[k][a]);
        hi[a] = std::max(hi[a], pos[k][a]);
      }
    const Vec3 gc = 0.5 * (lo + hi), gh = 0.5 * (hi - lo);
    GroupWork &w = work[g];
    w.cells.clear();
    w.bodies.clear();
    std::size_t stack[1024];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
      const std::size_t ni = stack[--sp];
      const auto &nd = nodes[ni];
      bool overlap = true;
      double d2 = 0.0;
      for (int a = 0; a < 3; ++a) {
        if (std::fabs(nd.center[a] - gc[a]) > nd.half * 1.0001 + gh[a]) overlap = false;
        const double d = std::fabs(nd.com[a] - gc[a]) - gh[a];
        if (d > 0) d2 += d * d;
      }
      const double crit = 2.0 * nd.half * inv_theta + nd.delta;
      if (!overlap && d2 > crit * crit) {
        if (nd.end - nd.begin <= kSmallCellAsBodies)
          w.add_bodies(nd.begin, nd.end);
        else
          w.cells.push_back(static_cast<kern::uint>(ni));
      } else if (nd.is_leaf()) {
        w.add_bodies(nd.begin, nd.end);
      } else {
        for (int c = 0; c < nd.n_children; ++c) stack[sp++] = nd.first_child + c;
      }
    }
  }

  // Concatenate into flat arrays.
  std::size_t ncell = 0, nbody = 0;
  for (const auto &w : work) {
    ncell += w.cells.size();
    nbody += w.bodies.size();
  }
  if (ncell > 0xffffffffu || nbody > 0xffffffffu)
    throw std::runtime_error("interaction lists too long for 32 bit offsets");
  out.cell_list.reserve(ncell);
  out.body_list.reserve(nbody);
  out.groups.resize(specs.size());
  for (std::size_t g = 0; g < specs.size(); ++g) {
    kern::GTGroup &grp = out.groups[g];
    grp = kern::GTGroup{};
    grp.begin = specs[g].begin;
    grp.end = specs[g].end;
    grp.cell_begin = static_cast<kern::uint>(out.cell_list.size());
    out.cell_list.insert(out.cell_list.end(), work[g].cells.begin(), work[g].cells.end());
    grp.cell_end = static_cast<kern::uint>(out.cell_list.size());
    grp.body_begin = static_cast<kern::uint>(out.body_list.size());
    out.body_list.insert(out.body_list.end(), work[g].bodies.begin(), work[g].bodies.end());
    grp.body_end = static_cast<kern::uint>(out.body_list.size());

    const std::uint64_t targets = grp.end - grp.begin;
    out.cell_interactions += targets * work[g].cells.size();
    out.body_interactions += targets * work[g].bodies.size() - targets;  // each skips itself
    out.max_group = std::max<int>(out.max_group, static_cast<int>(targets));
  }
}

}  // namespace galactic_tides
