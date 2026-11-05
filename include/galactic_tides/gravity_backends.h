#pragma once

// Alternative gravity solvers behind the ForceModel interface, including the
// Metal GPU ones, and a factory that picks one from run parameters.

#include <memory>
#include <string>
#include <vector>

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

// Builds the force model named by the parameters:
//   gravity = tree (default) | direct
//   device  = cpu (default) | metal
// and returns a one line description in *summary.
std::unique_ptr<ForceModel> make_gravity(const Params &params, const GravityParams &gp,
                                         std::string *summary = nullptr);

}  // namespace galactic_tides
