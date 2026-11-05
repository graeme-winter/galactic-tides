#include "galactic_tides/gravity_backends.h"

#include <algorithm>
#include <stdexcept>

#include "galactic_tides/gravity.h"
#include "galactic_tides/metal_backend.h"

namespace galactic_tides {

const char *device_name(Device d) { return d == Device::Metal ? "metal" : "cpu"; }

void pack_bodies(const std::vector<Vec3> &pos, const std::vector<double> &mass,
                 const Vec3 &origin, std::vector<kern::float4> &out) {
  out.resize(pos.size());
  for (std::size_t i = 0; i < pos.size(); ++i) {
    const Vec3 d = pos[i] - origin;
    out[i] = kern::float4{static_cast<float>(d.x), static_cast<float>(d.y),
                          static_cast<float>(d.z), static_cast<float>(mass[i])};
  }
}

kern::GTParams make_kernel_params(const GravityParams &gp, std::size_t n) {
  kern::GTParams p{};
  p.n = static_cast<kern::uint>(n);
  p.G = static_cast<float>(gp.G);
  p.eps2 = static_cast<float>(gp.eps * gp.eps);
  p.quadrupole = gp.quadrupole ? 1u : 0u;
  return p;
}

namespace {
Vec3 bounding_box_centre(const std::vector<Vec3> &pos) {
  if (pos.empty()) return {};
  Vec3 lo = pos[0], hi = pos[0];
  for (const auto &x : pos)
    for (int k = 0; k < 3; ++k) {
      lo[k] = std::min(lo[k], x[k]);
      hi[k] = std::max(hi[k], x[k]);
    }
  return 0.5 * (lo + hi);
}
}  // namespace

DirectGravity::DirectGravity(const GravityParams &gp, Device device) : gp_(gp), device_(device) {
  if (device_ == Device::Metal) ctx_ = std::make_unique<metal::Context>();
}

DirectGravity::~DirectGravity() = default;

void DirectGravity::compute(Particles &p) {
  const std::size_t n = p.size();
  stats_ = WalkStats{};
  stats_.particle_interactions = n > 0 ? n * (n - 1) : 0;
  if (device_ == Device::CPU) {
    compute_gravity_direct(p, gp_);
    return;
  }
  if (n > 0xffffffffu) throw std::runtime_error("too many particles for the Metal kernels");
  pack_bodies(p.pos, p.mass, bounding_box_centre(p.pos), bodies_);
  ctx_->direct(bodies_, make_kernel_params(gp_, n), out_);
  for (std::size_t i = 0; i < n; ++i) {
    p.acc[i] = Vec3(out_[i].x, out_[i].y, out_[i].z);
    p.pot[i] = out_[i].w;
  }
}

std::unique_ptr<ForceModel> make_gravity(const Params &params, const GravityParams &gp,
                                         std::string *summary) {
  const std::string kind = params.get_string("gravity", "tree");
  const std::string dev = params.get_string("device", "cpu");
  Device device;
  if (dev == "cpu") device = Device::CPU;
  else if (dev == "metal") device = Device::Metal;
  else throw std::runtime_error("device must be 'cpu' or 'metal', got '" + dev + "'");

  std::unique_ptr<ForceModel> model;
  std::string what;
  if (kind == "tree") {
    if (device != Device::CPU)
      throw std::runtime_error("gravity = tree runs on the CPU; use gravity = group for the GPU tree");
    const int leaf = static_cast<int>(params.get_int("leaf_size", 8));
    model = std::make_unique<TreeGravity>(gp, leaf);
    what = "tree (per-particle walk, double precision) on cpu";
  } else if (kind == "direct") {
    auto d = std::make_unique<DirectGravity>(gp, device);
    what = device == Device::CPU ? "direct summation, double precision, on cpu"
                                 : "direct summation, single precision, on metal";
    model = std::move(d);
  } else {
    throw std::runtime_error("gravity must be 'tree' or 'direct', got '" + kind + "'");
  }
  if (summary) *summary = what;
  return model;
}

}  // namespace galactic_tides
