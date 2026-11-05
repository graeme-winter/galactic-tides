// Used on builds without Metal: the API exists so that callers compile
// everywhere, and fails clearly if used.

#include <stdexcept>

#include "galactic_tides/metal_backend.h"

namespace galactic_tides {
namespace metal {

bool compiled_in() { return false; }

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
                   const gpu_vector<kern::uint> &, const gpu_vector<kern::GTRange> &,
                   const gpu_vector<kern::GTGroup> &, const kern::GTParams &, unsigned,
                   gpu_vector<kern::float4> &) {}
double Context::last_gpu_seconds() const { return 0.0; }
bool Context::last_zero_copy() const { return false; }

}  // namespace metal
}  // namespace galactic_tides
