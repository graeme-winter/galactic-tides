// Metal host code (Objective-C++, compiled with ARC). Kept deliberately
// small: bind arrays, dispatch, wait. All arithmetic is in
// include/galactic_tides/kernels/gravity_kernels.h, which the tests also run on
// the CPU, and the kernel wrappers in gravity_kernels.metal are compiled and
// run as C++ by test_metal_mock.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "galactic_tides/metal_backend.h"

namespace galactic_tides {
namespace metal {

static_assert(sizeof(kern::float4) == 16, "float4 must match Metal's 16 byte float4");
static_assert(sizeof(kern::GTParams) == 32, "GTParams layout");
static_assert(sizeof(kern::GTCell) == 48, "GTCell layout");
static_assert(sizeof(kern::GTGroup) == 32, "GTGroup layout");

bool compiled_in() { return true; }

namespace {

std::string describe(NSError *err) {
  if (err == nil) return "unknown error";
  return std::string([[err localizedDescription] UTF8String]);
}

// One kernel argument backed by a host array. Page-aligned arrays
// (gpu_vector) are wrapped in place with newBufferWithBytesNoCopy, and the
// wrapper is reused while the array keeps its storage; anything else falls
// back to a private shared buffer and a copy.
struct Binding {
  const void *wrapped = nullptr;  // host storage the buffer wraps, if no-copy
  size_t wrapped_bytes = 0;
  bool no_copy = false;
  id<MTLBuffer> buf = nil;
};

}  // namespace

void start_gpu_capture(const std::string &path) {
  std::string error;
  @autoreleasepool {
    MTLCaptureManager *manager = [MTLCaptureManager sharedCaptureManager];
    if (![manager supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
      error = "GPU capture to a file is not enabled: run with MTL_CAPTURE_ENABLED=1";
    } else {
      MTLCaptureDescriptor *descriptor = [[MTLCaptureDescriptor alloc] init];
      descriptor.captureObject = MTLCreateSystemDefaultDevice();
      descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
      descriptor.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
      NSError *err = nil;
      if (![manager startCaptureWithDescriptor:descriptor error:&err])
        error = "GPU capture failed to start: " + describe(err);
    }
  }
  if (!error.empty()) throw std::runtime_error(error);
}

void stop_gpu_capture() {
  @autoreleasepool {
    [[MTLCaptureManager sharedCaptureManager] stopCapture];
  }
}

struct Context::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> direct = nil;
  id<MTLComputePipelineState> tree = nil;
  id<MTLComputePipelineState> tree_simd = nil;
  unsigned simds = 4;
  const char *tree_kernel = "";
  Binding bodies, cells, cell_list, body_list, groups, out;
  bool allow_no_copy = true;  // switched off if Metal ever refuses to wrap
  double gpu_seconds = 0.0;
  bool zero_copy = false;

  void own_buffer(Binding &b, size_t bytes) {
    bytes = std::max<size_t>(bytes, 16);
    if (b.no_copy || b.buf == nil || [b.buf length] < bytes) {
      b.buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      if (b.buf == nil) throw std::runtime_error("Metal: failed to allocate a buffer");
    }
    b.no_copy = false;
    b.wrapped = nullptr;
    b.wrapped_bytes = 0;
  }

  // Try to make b wrap [ptr, ptr + alloc_bytes) without copying.
  bool wrap(Binding &b, const void *ptr, size_t alloc_bytes) {
    const size_t page = page_size();
    if (!allow_no_copy || ptr == nullptr || alloc_bytes == 0 ||
        reinterpret_cast<std::uintptr_t>(ptr) % page != 0 || alloc_bytes % page != 0)
      return false;
    if (b.no_copy && b.wrapped == ptr && b.wrapped_bytes == alloc_bytes) return true;
    id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:const_cast<void *>(ptr)
                                                  length:alloc_bytes
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
    if (buf == nil) {
      allow_no_copy = false;
      return false;
    }
    b.buf = buf;
    b.no_copy = true;
    b.wrapped = ptr;
    b.wrapped_bytes = alloc_bytes;
    return true;
  }

  template <class T>
  void bind_input(Binding &b, const gpu_vector<T> &v) {
    const size_t used = v.size() * sizeof(T);
    if (wrap(b, v.data(), allocation_bytes(v))) return;
    own_buffer(b, used);
    if (used) std::memcpy([b.buf contents], v.data(), used);
  }

  // The output must already have its final size.
  template <class T>
  void bind_output(Binding &b, gpu_vector<T> &v) {
    if (wrap(b, v.data(), allocation_bytes(v))) return;
    own_buffer(b, v.size() * sizeof(T));
  }

  template <class T>
  void collect_output(Binding &b, gpu_vector<T> &v) {
    if (!b.no_copy && !v.empty()) std::memcpy(v.data(), [b.buf contents], v.size() * sizeof(T));
  }

  // Commit, wait, record GPU time, and turn a GPU-side failure into a message.
  std::string finish(id<MTLCommandBuffer> cb, const char *what) {
    [cb commit];
    [cb waitUntilCompleted];
    if ([cb status] == MTLCommandBufferStatusError)
      return std::string("Metal: ") + what + " failed: " + describe([cb error]);
    gpu_seconds = [cb GPUEndTime] - [cb GPUStartTime];
    return {};
  }
};

Context::Context() : impl_(new Impl) {
  std::string error;
  @autoreleasepool {
    Impl &m = *impl_;
    m.device = MTLCreateSystemDefaultDevice();
    if (m.device == nil) {
      error = "Metal: no GPU device available";
    } else {
      m.queue = [m.device newCommandQueue];
      NSError *err = nil;
      MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
      NSString *source = [NSString stringWithUTF8String:kKernelSource];
      id<MTLLibrary> library = [m.device newLibraryWithSource:source options:options error:&err];
      if (library == nil) {
        error = "Metal: kernel compilation failed:\n" + describe(err);
      } else {
        auto pipeline = [&](const char *name) -> id<MTLComputePipelineState> {
          id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
          if (fn == nil) {
            error = std::string("Metal: kernel not found: ") + name;
            return nil;
          }
          NSError *perr = nil;
          id<MTLComputePipelineState> pso = [m.device newComputePipelineStateWithFunction:fn
                                                                                    error:&perr];
          if (pso == nil) error = std::string("Metal: pipeline for ") + name + ": " + describe(perr);
          return pso;
        };
        m.direct = pipeline("gt_direct");
        if (error.empty()) m.tree = pipeline("gt_tree");
        if (error.empty()) m.tree_simd = pipeline("gt_tree_simd");
      }
    }
  }
  if (!error.empty()) throw std::runtime_error(error);
}

Context::~Context() = default;

std::string Context::device_name() const {
  std::string name;
  @autoreleasepool {
    name = [[impl_->device name] UTF8String];
  }
  return name;
}

void Context::set_tree_simds(unsigned simds) { impl_->simds = simds; }
const char *Context::last_tree_kernel() const { return impl_->tree_kernel; }
double Context::last_gpu_seconds() const { return impl_->gpu_seconds; }
bool Context::last_zero_copy() const { return impl_->zero_copy; }

void Context::direct(const gpu_vector<kern::float4> &bodies, const kern::GTParams &p,
                     gpu_vector<kern::float4> &out) {
  const size_t n = bodies.size();
  out.resize(n);
  impl_->gpu_seconds = 0.0;
  if (n == 0) return;
  std::string error;
  @autoreleasepool {
    try {
      Impl &m = *impl_;
      m.bind_input(m.bodies, bodies);
      m.bind_output(m.out, out);
      m.zero_copy = m.bodies.no_copy && m.out.no_copy;
      // The kernel's tile holds 256 bodies, so threadgroups must not be larger.
      const NSUInteger tg = std::min<NSUInteger>(256, [m.direct maxTotalThreadsPerThreadgroup]);
      const NSUInteger groups = (n + tg - 1) / tg;
      id<MTLCommandBuffer> cb = [m.queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:m.direct];
      [enc setBuffer:m.bodies.buf offset:0 atIndex:0];
      [enc setBuffer:m.out.buf offset:0 atIndex:1];
      [enc setBytes:&p length:sizeof(p) atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [enc endEncoding];
      error = m.finish(cb, "direct kernel");
      if (error.empty()) m.collect_output(m.out, out);
    } catch (const std::exception &e) {
      error = e.what();  // rethrown outside the autorelease pool
    }
  }
  if (!error.empty()) throw std::runtime_error(error);
}

void Context::tree(const gpu_vector<kern::float4> &bodies, const gpu_vector<kern::GTCell> &cells,
                   const gpu_vector<kern::uint> &cell_list,
                   const gpu_vector<kern::uint> &body_list,
                   const gpu_vector<kern::GTGroup> &groups, const kern::GTParams &p,
                   unsigned threads_per_group, gpu_vector<kern::float4> &out) {
  const size_t n = bodies.size();
  out.resize(n);
  impl_->gpu_seconds = 0.0;
  if (n == 0 || groups.empty()) return;
  if (p.n != n || p.n_groups != groups.size())
    throw std::invalid_argument("Metal tree: parameters do not match the arrays");
  std::string error;
  @autoreleasepool {
    try {
      Impl &m = *impl_;
      const bool use_simd =
          m.simds > 0 && threads_per_group <= GT_SIMD_WIDTH &&
          [m.tree_simd threadExecutionWidth] == GT_SIMD_WIDTH &&
          m.simds * GT_SIMD_WIDTH <= [m.tree_simd maxTotalThreadsPerThreadgroup];
      const NSUInteger limit = std::min<NSUInteger>(GT_TREE_TILE, [m.tree maxTotalThreadsPerThreadgroup]);
      if (!use_simd && (threads_per_group == 0 || threads_per_group > limit)) {
        error = "Metal tree: " + std::to_string(threads_per_group) +
                " threads per group, but at most " + std::to_string((unsigned long)limit) +
                " are possible here; use a smaller group_size";
      } else {
        m.bind_input(m.bodies, bodies);
        m.bind_input(m.cells, cells);
        m.bind_input(m.cell_list, cell_list);
        m.bind_input(m.body_list, body_list);
        m.bind_input(m.groups, groups);
        m.bind_output(m.out, out);
        m.zero_copy = m.bodies.no_copy && m.cells.no_copy && m.cell_list.no_copy &&
                      m.body_list.no_copy && m.groups.no_copy && m.out.no_copy;
        id<MTLCommandBuffer> cb = [m.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:(use_simd ? m.tree_simd : m.tree)];
        [enc setBuffer:m.bodies.buf offset:0 atIndex:0];
        [enc setBuffer:m.cells.buf offset:0 atIndex:1];
        [enc setBuffer:m.cell_list.buf offset:0 atIndex:2];
        [enc setBuffer:m.body_list.buf offset:0 atIndex:3];
        [enc setBuffer:m.groups.buf offset:0 atIndex:4];
        [enc setBuffer:m.out.buf offset:0 atIndex:5];
        [enc setBytes:&p length:sizeof(p) atIndex:6];
        if (use_simd) {
          const NSUInteger threadgroups = (groups.size() + m.simds - 1) / m.simds;
          [enc dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(m.simds * GT_SIMD_WIDTH, 1, 1)];
        } else {
          [enc dispatchThreadgroups:MTLSizeMake(groups.size(), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
        }
        [enc endEncoding];
        m.tree_kernel = use_simd ? "simd" : "tiled";
        error = m.finish(cb, "tree kernel");
        if (error.empty()) m.collect_output(m.out, out);
      }
    } catch (const std::exception &e) {
      error = e.what();  // rethrown outside the autorelease pool
    }
  }
  if (!error.empty()) throw std::runtime_error(error);
}

}  // namespace metal
}  // namespace galactic_tides
