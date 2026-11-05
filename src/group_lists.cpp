#include "galactic_tides/group_lists.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace galactic_tides {

namespace {

struct GroupSpec {
  std::uint32_t begin, end;  // targets, tree order
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

  out.max_group = 0;
  out.max_list = 0;
  out.cell_interactions = out.body_interactions = 0;
  out.bodies.resize(n);
  out.cells.resize(nodes.size());
  if (n == 0) {
    out.groups.clear();
    out.cell_list.clear();
    out.body_list.clear();
    return;
  }

  // Bodies and cells relative to the root centre, in single precision.
  out.origin = nodes[0].center;
  const double inv_theta = 1.0 / gp.theta;
  out.scratch_nodes.resize(nodes.size());
#pragma omp parallel
  {
#pragma omp for schedule(static) nowait
    for (long k = 0; k < static_cast<long>(n); ++k) {
      const Vec3 d = pos[k] - out.origin;
      out.bodies[k] = kern::float4{float(d.x), float(d.y), float(d.z), float(mass[k])};
    }
#pragma omp for schedule(static)
    for (long k = 0; k < static_cast<long>(nodes.size()); ++k) {
      const auto &nd = nodes[k];
      const Vec3 c = nd.com - out.origin;
      const double *q = nd.quad;
      out.cells[k].com_mass = kern::float4{float(c.x), float(c.y), float(c.z), float(nd.mass)};
      out.cells[k].q0 = kern::float4{float(q[0]), float(q[1]), float(q[2]), float(q[3])};
      out.cells[k].q1 = kern::float4{float(q[4]), float(q[5]), 0.0f, 0.0f};
      out.scratch_nodes[k] = GroupWalkNode{nd.center, nd.half, nd.com,
                                           2.0 * nd.half * inv_theta + nd.delta,
                                           nd.first_child, nd.n_children, nd.begin, nd.end};
    }
  }

  // Choose groups: the highest nodes holding at most group_size bodies. A
  // leaf can hold more (coincident particles at maximum depth, or a leaf
  // size above the group size); such leaves are cut into chunks.
  //
  // With pack, consecutive small children of the same node are merged while
  // they fit, so a packed group always lies inside its parent's cube.
  // (Packing any neighbours in tree order is not safe: two nodes adjacent in
  // tree order can be on opposite sides of the system, and a group spanning
  // them opens most of the tree. On an M4 Max one such group, with a list of
  // ~187000 entries, set the GPU kernel's time.)
  std::vector<GroupSpec> specs;
  const std::uint32_t gs = static_cast<std::uint32_t>(group_size);
  auto chunk = [&](std::uint32_t begin, std::uint32_t end) {
    for (std::uint32_t c = begin; c < end; c += gs)
      specs.push_back({c, std::min<std::uint32_t>(end, c + gs)});
  };
  // Depth first, children in order, so groups come out in tree order.
  std::function<void(std::size_t)> visit = [&](std::size_t ni) {
    const auto &nd = nodes[ni];
    if (nd.end - nd.begin <= gs || nd.is_leaf()) {
      chunk(nd.begin, nd.end);
      return;
    }
    GroupSpec pending{0, 0};
    bool have_pending = false;
    auto flush = [&] {
      if (have_pending) specs.push_back(pending);
      have_pending = false;
    };
    for (int c = 0; c < nd.n_children; ++c) {
      const std::size_t ci = static_cast<std::size_t>(nd.first_child + c);
      const auto &ch = nodes[ci];
      const std::uint32_t cc = ch.end - ch.begin;
      if (cc <= gs) {
        if (pack && have_pending && (pending.end - pending.begin) + cc <= gs) {
          pending.end = ch.end;  // siblings are contiguous in tree order
        } else {
          flush();
          pending = {ch.begin, ch.end};
          have_pending = true;
        }
      } else {
        flush();
        visit(ci);
      }
    }
    flush();
  };
  visit(0);

  // Walk once per group, into per-group scratch lists kept between builds.
  const std::size_t ng = specs.size();
  if (out.scratch_cells.size() < ng) {
    out.scratch_cells.resize(ng);
    out.scratch_bodies.resize(ng);
  }
  const GroupWalkNode *wn = out.scratch_nodes.data();
#pragma omp parallel for schedule(dynamic, 16)
  for (long g = 0; g < static_cast<long>(ng); ++g) {
    const GroupSpec &spec = specs[g];
    Vec3 lo = pos[spec.begin], hi = lo;
    for (std::uint32_t k = spec.begin; k < spec.end; ++k)
      for (int a = 0; a < 3; ++a) {
        lo[a] = std::min(lo[a], pos[k][a]);
        hi[a] = std::max(hi[a], pos[k][a]);
      }
    const Vec3 gc = 0.5 * (lo + hi), gh = 0.5 * (hi - lo);
    std::vector<kern::uint> &cells = out.scratch_cells[g];
    std::vector<kern::uint> &bodies = out.scratch_bodies[g];
    cells.clear();
    bodies.clear();
    auto add_bodies = [&](std::uint32_t begin, std::uint32_t end) {
      const std::size_t at = bodies.size();
      bodies.resize(at + (end - begin));
      for (std::uint32_t j = begin; j < end; ++j) bodies[at + (j - begin)] = j;
    };
    std::uint32_t stack[1024];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
      const std::uint32_t ni = stack[--sp];
      const GroupWalkNode &nd = wn[ni];
      bool overlap = true;
      double d2 = 0.0;
      for (int a = 0; a < 3; ++a) {
        if (std::fabs(nd.center[a] - gc[a]) > nd.half * 1.0001 + gh[a]) overlap = false;
        const double d = std::fabs(nd.com[a] - gc[a]) - gh[a];
        if (d > 0) d2 += d * d;
      }
      if (!overlap && d2 > nd.crit * nd.crit) {
        if (nd.end - nd.begin <= kSmallCellAsBodies)
          add_bodies(nd.begin, nd.end);
        else
          cells.push_back(ni);
      } else if (nd.n_children == 0) {
        add_bodies(nd.begin, nd.end);
      } else {
        for (int c = 0; c < nd.n_children; ++c)
          stack[sp++] = static_cast<std::uint32_t>(nd.first_child + c);
      }
    }
  }

  // Concatenate into the flat arrays: offsets serially, copies in parallel.
  std::uint64_t ncell = 0, nbody = 0;
  out.groups.resize(ng);
  for (std::size_t g = 0; g < ng; ++g) {
    kern::GTGroup &grp = out.groups[g];
    grp = kern::GTGroup{};
    grp.begin = specs[g].begin;
    grp.end = specs[g].end;
    grp.cell_begin = static_cast<kern::uint>(ncell);
    ncell += out.scratch_cells[g].size();
    grp.cell_end = static_cast<kern::uint>(ncell);
    grp.body_begin = static_cast<kern::uint>(nbody);
    nbody += out.scratch_bodies[g].size();
    grp.body_end = static_cast<kern::uint>(nbody);
    if (ncell > 0xffffffffu || nbody > 0xffffffffu)
      throw std::runtime_error("interaction lists too long for 32 bit offsets");
  }
  out.cell_list.resize(ncell);
  out.body_list.resize(nbody);
  std::uint64_t cell_int = 0, body_int = 0, max_list = 0;
  int max_group = 0;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : cell_int, body_int) \
    reduction(max : max_list, max_group)
  for (long g = 0; g < static_cast<long>(ng); ++g) {
    const kern::GTGroup &grp = out.groups[g];
    const auto &cells = out.scratch_cells[g];
    const auto &bodies = out.scratch_bodies[g];
    std::copy(cells.begin(), cells.end(), out.cell_list.begin() + grp.cell_begin);
    std::copy(bodies.begin(), bodies.end(), out.body_list.begin() + grp.body_begin);
    const std::uint64_t targets = grp.end - grp.begin;
    cell_int += targets * cells.size();
    body_int += targets * bodies.size() - targets;  // each target skips itself
    max_list = std::max<std::uint64_t>(max_list, cells.size() + bodies.size());
    max_group = std::max(max_group, static_cast<int>(targets));
  }
  out.cell_interactions = cell_int;
  out.body_interactions = body_int;
  out.max_list = max_list;
  out.max_group = max_group;
}

}  // namespace galactic_tides
