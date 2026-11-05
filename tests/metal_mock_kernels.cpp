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
// Barrier for the threads of one threadgroup. Metal requires every thread of
// a threadgroup to reach every threadgroup_barrier; a thread that returns
// early while others still wait would hang or misbehave on a GPU. Here it is
// recorded as a violation and the waiting threads are released, so the test
// fails instead of hanging.
class Barrier {
 public:
  void reset(unsigned count) {
    std::lock_guard<std::mutex> lock(m_);
    count_ = count;
    waiting_ = 0;
    exited_ = false;
  }
  void wait() {
    std::unique_lock<std::mutex> lock(m_);
    if (exited_) {
      violation_ = true;
      return;
    }
    const unsigned gen = generation_;
    if (++waiting_ == count_) {
      waiting_ = 0;
      ++generation_;
      cv_.notify_all();
      return;
    }
    cv_.wait(lock, [&] { return gen != generation_ || exited_; });
    if (gen == generation_) {  // released because a thread left early
      violation_ = true;
      --waiting_;
    }
  }
  // A thread finished the kernel.
  void exit_thread() {
    std::lock_guard<std::mutex> lock(m_);
    exited_ = true;
    if (waiting_ > 0) {
      violation_ = true;
      cv_.notify_all();
    }
  }
  bool violation() const { return violation_; }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  unsigned count_ = 1, waiting_ = 0, generation_ = 0;
  bool exited_ = false;
  bool violation_ = false;
};
Barrier g_barrier;

// SIMD groups: each has its own barrier and exchange slots, and each thread
// knows which SIMD group and lane it is.
constexpr unsigned kMaxSimds = 32, kSimdWidth = 32;
Barrier g_simd_barrier[kMaxSimds];
thread_local unsigned tl_simd = 0, tl_lane = 0;
bool g_simd_violation = false;
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

template <class T>
T shuffle(T value, unsigned short lane) {
  static T slots[kMaxSimds][kSimdWidth];
  if (lane >= kSimdWidth) g_simd_violation = true;
  slots[tl_simd][tl_lane] = value;
  g_simd_barrier[tl_simd].wait();  // every lane has published
  const T result = slots[tl_simd][lane % kSimdWidth];
  g_simd_barrier[tl_simd].wait();  // every lane has read before slots are reused
  return result;
}
float4 simd_shuffle(float4 value, unsigned short lane) { return shuffle(value, lane); }
unsigned int simd_shuffle(unsigned int value, unsigned short lane) { return shuffle(value, lane); }
}  // namespace metal

namespace mock {

static_assert(sizeof(::GTParams) == 32 && sizeof(::GTCell) == 48 && sizeof(::GTGroup) == 32 &&
                  sizeof(metal::float4) == 16,
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
        g_barrier.exit_thread();
      });
    for (auto &t : threads) t.join();
  }
}

void run_tree(const void *bodies, const void *cells, const void *cell_list, const void *body_list,
              const void *groups, const void *params, unsigned threads_per_group, void *out) {
  ::GTParams p = *static_cast<const ::GTParams *>(params);
  for (unsigned gid = 0; gid < p.n_groups; ++gid) {
    g_barrier.reset(threads_per_group);
    std::vector<std::thread> threads;
    for (unsigned lid = 0; lid < threads_per_group; ++lid)
      threads.emplace_back([&, lid] {
        gt_tree(static_cast<const metal::float4 *>(bodies), static_cast<const ::GTCell *>(cells),
                static_cast<const unsigned *>(cell_list), static_cast<const unsigned *>(body_list),
                static_cast<const ::GTGroup *>(groups), static_cast<metal::float4 *>(out), p, gid,
                lid, threads_per_group);
        g_barrier.exit_thread();
      });
    for (auto &t : threads) t.join();
  }
}

void run_tree_simd(const void *bodies, const void *cells, const void *cell_list,
                   const void *body_list, const void *groups, const void *params,
                   unsigned simds_per_threadgroup, void *out) {
  ::GTParams p = *static_cast<const ::GTParams *>(params);
  const unsigned simds = simds_per_threadgroup;
  const unsigned threadgroups = (p.n_groups + simds - 1) / simds;
  for (unsigned gid = 0; gid < threadgroups; ++gid) {
    for (unsigned s = 0; s < simds; ++s) g_simd_barrier[s].reset(kSimdWidth);
    std::vector<std::thread> threads;
    for (unsigned s = 0; s < simds; ++s)
      for (unsigned lane = 0; lane < kSimdWidth; ++lane)
        threads.emplace_back([&, s, lane] {
          tl_simd = s;
          tl_lane = lane;
          gt_tree_simd(static_cast<const metal::float4 *>(bodies),
                       static_cast<const ::GTCell *>(cells), static_cast<const unsigned *>(cell_list),
                       static_cast<const unsigned *>(body_list),
                       static_cast<const ::GTGroup *>(groups), static_cast<metal::float4 *>(out), p,
                       gid, s, simds, lane);
          g_simd_barrier[s].exit_thread();
        });
    for (auto &t : threads) t.join();
  }
}

bool barrier_violation() {
  bool v = g_barrier.violation() || g_simd_violation;
  for (const auto &b : g_simd_barrier) v = v || b.violation();
  return v;
}

}  // namespace mock
