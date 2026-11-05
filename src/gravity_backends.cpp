#include "galactic_tides/gravity_backends.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

#include "galactic_tides/gravity.h"
#include "galactic_tides/metal_backend.h"

namespace galactic_tides {

const char *device_name(Device d) { return d == Device::Metal ? "metal" : "cpu"; }

void pack_bodies(const std::vector<Vec3> &pos, const std::vector<double> &mass,
                 const Vec3 &origin, gpu_vector<kern::float4> &out) {
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

void evaluate_group_lists_cpu(const GroupLists &lists, const kern::GTParams &p,
                              gpu_vector<kern::float4> &out) {
  out.resize(lists.bodies.size());
  const long ng = static_cast<long>(lists.groups.size());
#pragma omp parallel for schedule(dynamic, 8)
  for (long g = 0; g < ng; ++g) {
    const kern::GTGroup grp = lists.groups[g];
    for (kern::uint i = grp.begin; i < grp.end; ++i) {
      const kern::GTAcc a =
          kern::gt_group_accel(i, grp, lists.bodies.data(), lists.cells.data(),
                               lists.cell_list.data(), lists.body_list.data(), p);
      out[i] = kern::float4{a.ax, a.ay, a.az, a.pot};
    }
  }
}

GroupTreeGravity::GroupTreeGravity(const GravityParams &gp, Device device, int group_size,
                                   int leaf_size, int pack)
    : gp_(gp),
      device_(device),
      group_size_(group_size),
      leaf_size_(leaf_size),
      pack_(pack < 0 ? device == Device::Metal : pack != 0) {
  if (group_size_ < 1) throw std::invalid_argument("group_size must be at least 1");
  if (device_ == Device::Metal && group_size_ > static_cast<int>(GT_TREE_TILE))
    throw std::invalid_argument("group_size can be at most " + std::to_string(GT_TREE_TILE) +
                                " on Metal");
  if (device_ == Device::Metal) ctx_ = std::make_unique<metal::Context>();
}

GroupTreeGravity::~GroupTreeGravity() = default;

WalkStats GroupTreeGravity::stats() const {
  WalkStats s;
  s.cell_interactions = lists_.cell_interactions;
  s.particle_interactions = lists_.body_interactions;
  return s;
}

void GroupTreeGravity::compute(Particles &p) {
  using clock = std::chrono::steady_clock;
  auto secs = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
  };
  const auto t0 = clock::now();
  tree_.build(p.pos, p.mass, leaf_size_);
  const auto t1 = clock::now();
  build_group_lists(tree_, gp_, group_size_, lists_, pack_);
  const auto t2 = clock::now();
  kern::GTParams kp = make_kernel_params(gp_, lists_.bodies.size());
  kp.n_groups = static_cast<kern::uint>(lists_.groups.size());
  if (device_ == Device::Metal)
    ctx_->tree(lists_.bodies, lists_.cells, lists_.cell_list, lists_.body_list, lists_.groups, kp,
               static_cast<unsigned>(std::max(lists_.max_group, 1)), out_);
  else
    evaluate_group_lists_cpu(lists_, kp, out_);
  const auto t3 = clock::now();
  timings_ = {secs(t0, t1), secs(t1, t2), secs(t2, t3), 0.0, false};
  if (device_ == Device::Metal) {
    timings_.gpu = ctx_->last_gpu_seconds();
    timings_.zero_copy = ctx_->last_zero_copy();
  }
  const auto &order = tree_.order();
  for (std::size_t k = 0; k < order.size(); ++k) {
    p.acc[order[k]] = Vec3(out_[k].x, out_[k].y, out_[k].z);
    p.pot[order[k]] = out_[k].w;
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
  } else if (kind == "group") {
    const int group = static_cast<int>(params.get_int("group_size", 32));
    const int leaf = static_cast<int>(params.get_int("leaf_size", 8));
    const bool pack = params.get_bool("group_pack", device == Device::Metal);
    model = std::make_unique<GroupTreeGravity>(gp, device, group, leaf, pack ? 1 : 0);
    what = std::string("tree with grouped walk (groups of up to ") + std::to_string(group) +
           (pack ? ", packed" : "") + "), single precision lists on " + device_name(device);
  } else if (kind == "direct") {
    auto d = std::make_unique<DirectGravity>(gp, device);
    what = device == Device::CPU ? "direct summation, double precision, on cpu"
                                 : "direct summation, single precision, on metal";
    model = std::move(d);
  } else {
    throw std::runtime_error("gravity must be 'tree', 'group' or 'direct', got '" + kind + "'");
  }
  if (summary) *summary = what;
  return model;
}

}  // namespace galactic_tides
