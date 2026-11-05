// Used on builds without Metal: the API exists so that callers compile
// everywhere, and fails clearly if used.

#include <stdexcept>

#include "galactic_tides/metal_backend.h"

namespace galactic_tides {
namespace metal {

bool compiled_in() { return false; }

void start_gpu_capture(const std::string &) {
  throw std::runtime_error("GPU capture needs a build with Metal");
}
void stop_gpu_capture() {}

struct Context::Impl {};

Context::Context() {
  throw std::runtime_error(
      "this build of galactic-tides has no Metal support (it needs macOS; configure with "
      "-DGALACTIC_TIDES_USE_METAL=ON)");
}
Context::~Context() = default;
std::string Context::device_name() const { return {}; }
void Context::direct(const gpu_vector<kern::float4> &, const kern::GTParams &,
                     gpu_vector<kern::float4> &) {}
void Context::tree(const gpu_vector<kern::float4> &, const gpu_vector<kern::GTCell> &,
                   const gpu_vector<kern::uint> &, const gpu_vector<kern::uint> &,
                   const gpu_vector<kern::GTGroup> &, const kern::GTParams &, unsigned,
                   gpu_vector<kern::float4> &) {}
void Context::set_tree_simds(unsigned) {}
void Context::set_dispatch_repeats(unsigned) {}
const char *Context::last_tree_kernel() const { return ""; }
double Context::last_gpu_seconds() const { return 0.0; }
bool Context::last_zero_copy() const { return false; }

}  // namespace metal
}  // namespace galactic_tides
