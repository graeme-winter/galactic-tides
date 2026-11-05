#pragma once

// Grouped tree walk (Barnes 1990): instead of walking the tree once per
// particle, walk it once per group of nearby particles (a tree node with at
// most group_size bodies) and record which cells and leaves that group
// interacts with. Every target in the group then evaluates the same list,
// which is what a GPU wants: uniform work across a SIMD group and the CPU
// does ~group_size times fewer walks.
//
// The opening test is the per-particle one applied to the group's bounding
// box: a cell is accepted only if its centre of mass is further than
// l/theta + delta from the nearest point of the box and the cell does not
// overlap the box, so every target in the group is at least as well served
// as by its own walk. Opened leaves, and accepted cells with very few bodies
// (exact, and cheaper than a quadrupole), contribute their bodies; bodies
// are listed one by one so the GPU can load them in parallel.

#include <cstdint>
#include <vector>

#include "galactic_tides/kernels/gravity_kernels.h"
#include "galactic_tides/octree.h"
#include "galactic_tides/page_allocator.h"

namespace galactic_tides {

struct GroupLists {
  Vec3 origin;                         // bodies and cells are relative to this
  // Page-aligned so the GPU can use them in place.
  gpu_vector<kern::float4> bodies;     // tree order
  gpu_vector<kern::GTCell> cells;      // one per tree node
  gpu_vector<kern::uint> cell_list;    // node indices
  gpu_vector<kern::uint> body_list;    // body indices (tree order)
  gpu_vector<kern::GTGroup> groups;    // contiguous, in tree order
  int max_group = 0;                   // largest group (threads per threadgroup)
  std::uint64_t cell_interactions = 0;  // summed over targets
  std::uint64_t body_interactions = 0;
};

// Cells with at most this many bodies are listed as bodies when accepted.
constexpr std::uint32_t kSmallCellAsBodies = 2;

// With pack (the default), neighbouring small groups are merged, in tree
// order, while they fit in group_size: the octree's natural groups are often
// far from full, and on a GPU every unused thread in a SIMD group is wasted.
void build_group_lists(const Octree &tree, const GravityParams &gp, int group_size,
                       GroupLists &out, bool pack = true);

}  // namespace galactic_tides
