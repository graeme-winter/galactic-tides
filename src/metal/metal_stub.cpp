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
void Context::direct(const std::vector<kern::float4> &, const kern::GTParams &,
                     std::vector<kern::float4> &) {}
void Context::tree(const std::vector<kern::float4> &, const std::vector<kern::GTCell> &,
                   const std::vector<kern::uint> &, const std::vector<kern::GTRange> &,
                   const std::vector<kern::GTGroup> &, const kern::GTParams &, unsigned,
                   std::vector<kern::float4> &) {}

}  // namespace metal
}  // namespace galactic_tides
