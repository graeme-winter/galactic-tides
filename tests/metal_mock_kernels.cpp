// Compiles src/metal/gravity_kernels.metal (and the shared header in its
// Metal branch) as C++ against tests/mock_metal/metal_stdlib, and runs the
// kernels the way a GPU would: one CPU thread per GPU thread, threadgroups
// one after another, threadgroup memory shared by the group's threads and
// threadgroup_barrier a real barrier. This exercises the actual kernel
// source, tiling and index arithmetic included, on any machine.
//
// This translation unit must not include other galactic-tides headers: it sees
// the kernel structs in their Metal form, in the global namespace.

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

#include "metal_mock_kernels.h"

namespace {
// Reusable barrier for the threads of one threadgroup.
class Barrier {
 public:
  void reset(unsigned count) {
    count_ = count;
    waiting_ = 0;
  }
  void wait() {
    std::unique_lock<std::mutex> lock(m_);
    const unsigned gen = generation_;
    if (++waiting_ == count_) {
      waiting_ = 0;
      ++generation_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [&] { return gen != generation_; });
    }
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  unsigned count_ = 1, waiting_ = 0, generation_ = 0;
};
Barrier g_barrier;
}  // namespace

// Metal address spaces and kernel qualifiers, mapped onto C++. threadgroup
// variables become function statics: shared by every thread of the group
// being run, which is exactly their Metal meaning since groups run one at a
// time here.
#define __METAL_VERSION__ 300
#define kernel
#define device
#define constant
#define threadgroup static

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wattributes"  // [[buffer(0)]] and friends
#endif

#include "gravity_kernels.metal"

namespace metal {
void threadgroup_barrier(mem_flags) { g_barrier.wait(); }
}  // namespace metal

namespace mock {

static_assert(sizeof(::GTParams) == 32 && sizeof(::GTCell) == 48 && sizeof(::GTGroup) == 32 &&
                  sizeof(::GTRange) == 8 && sizeof(metal::float4) == 16,
              "mock Metal layouts");

void run_direct(const void *bodies, const void *params, unsigned threads_per_group, void *out) {
  const auto *b = static_cast<const metal::float4 *>(bodies);
  const ::GTParams p = *static_cast<const ::GTParams *>(params);
  ::GTParams p_copy = p;
  auto *o = static_cast<metal::float4 *>(out);
  const unsigned groups = (p.n + threads_per_group - 1) / threads_per_group;
  for (unsigned g = 0; g < groups; ++g) {
    g_barrier.reset(threads_per_group);
    std::vector<std::thread> threads;
    for (unsigned lid = 0; lid < threads_per_group; ++lid)
      threads.emplace_back([&, lid] {
        gt_direct(b, o, p_copy, g * threads_per_group + lid, lid, threads_per_group);
      });
    for (auto &t : threads) t.join();
  }
}

void run_tree(const void *bodies, const void *cells, const void *cell_list, const void *leaf_list,
              const void *groups, const void *params, unsigned threads_per_group, void *out) {
  ::GTParams p = *static_cast<const ::GTParams *>(params);
  for (unsigned gid = 0; gid < p.n_groups; ++gid)
    for (unsigned lid = 0; lid < threads_per_group; ++lid)
      gt_tree(static_cast<const metal::float4 *>(bodies), static_cast<const ::GTCell *>(cells),
              static_cast<const unsigned *>(cell_list), static_cast<const ::GTRange *>(leaf_list),
              static_cast<const ::GTGroup *>(groups), static_cast<metal::float4 *>(out), p, gid,
              lid);
}

}  // namespace mock
