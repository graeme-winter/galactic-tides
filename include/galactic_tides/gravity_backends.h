#pragma once

// Alternative gravity solvers behind the ForceModel interface, including the
// Metal GPU ones, and a factory that picks one from run parameters.

#include <memory>
#include <string>
#include <vector>

#include "galactic_tides/group_lists.h"
#include "galactic_tides/integrator.h"
#include "galactic_tides/kernels/gravity_kernels.h"
#include "galactic_tides/params.h"

namespace galactic_tides {

namespace metal {
class Context;
}

enum class Device { CPU, Metal };
const char *device_name(Device d);

// Positions relative to `origin`, in single precision, with mass in w: the
// layout every kernel reads. Working relative to the middle of the system
// keeps single precision differences accurate.
void pack_bodies(const std::vector<Vec3> &pos, const std::vector<double> &mass,
                 const Vec3 &origin, std::vector<kern::float4> &out);
kern::GTParams make_kernel_params(const GravityParams &gp, std::size_t n);

// O(N^2) summation. On the CPU this is the double precision reference
// (compute_gravity_direct); on Metal it is the single precision tiled kernel.
class DirectGravity : public ForceModel {
 public:
  DirectGravity(const GravityParams &gp, Device device);
  ~DirectGravity() override;
  void compute(Particles &p) override;
  WalkStats stats() const override { return stats_; }

 private:
  GravityParams gp_;
  Device device_;
  std::unique_ptr<metal::Context> ctx_;
  std::vector<kern::float4> bodies_, out_;
  WalkStats stats_;
};

// Barnes-Hut with a grouped walk: the tree and the interaction lists are
// built on the CPU (in double precision), and the lists are evaluated by the
// single precision kernels, either on the CPU (OpenMP over groups, the same
// code the GPU runs) or on the GPU.
class GroupTreeGravity : public ForceModel {
 public:
  struct Timings {
    double tree = 0, lists = 0, evaluate = 0;  // seconds, last compute()
  };
  // pack < 0 means the device's default: packed groups on Metal (fills the
  // SIMD lanes), unpacked on the CPU (shorter lists, no lanes to fill).
  GroupTreeGravity(const GravityParams &gp, Device device, int group_size = 32,
                   int leaf_size = 8, int pack = -1);
  ~GroupTreeGravity() override;
  void compute(Particles &p) override;
  WalkStats stats() const override;
  const Timings &timings() const { return timings_; }
  const GroupLists &lists() const { return lists_; }
  const Octree &tree() const { return tree_; }

 private:
  GravityParams gp_;
  Device device_;
  int group_size_, leaf_size_;
  bool pack_;
  Octree tree_;
  GroupLists lists_;
  std::vector<kern::float4> out_;
  std::unique_ptr<metal::Context> ctx_;
  Timings timings_;
};

// Evaluate grouped lists on the CPU with the shared kernels: out[i] in tree
// order. Exposed for tests.
void evaluate_group_lists_cpu(const GroupLists &lists, const kern::GTParams &p,
                              std::vector<kern::float4> &out);

// Builds the force model named by the parameters:
//   gravity = tree (default) | group | direct
//   device  = cpu (default) | metal
//   group_size (group only, default 32), group_pack (group only, default:
//   on for metal, off for cpu), leaf_size (tree and group, default 8)
// and returns a one line description in *summary.
std::unique_ptr<ForceModel> make_gravity(const Params &params, const GravityParams &gp,
                                         std::string *summary = nullptr);

}  // namespace galactic_tides
