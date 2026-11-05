// Metal host code (Objective-C++, compiled with ARC). Kept deliberately
// small: upload arrays, dispatch, wait, download. All arithmetic is in
// include/galactic_tides/kernels/gravity_kernels.h, which the tests also run on
// the CPU.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "galactic_tides/metal_backend.h"

namespace galactic_tides {
namespace metal {

static_assert(sizeof(kern::float4) == 16, "float4 must match Metal's 16 byte float4");
static_assert(sizeof(kern::GTParams) == 32, "GTParams layout");
static_assert(sizeof(kern::GTCell) == 48, "GTCell layout");
static_assert(sizeof(kern::GTRange) == 8, "GTRange layout");
static_assert(sizeof(kern::GTGroup) == 32, "GTGroup layout");

bool compiled_in() { return true; }

namespace {

std::string describe(NSError *err) {
  if (err == nil) return "unknown error";
  return std::string([[err localizedDescription] UTF8String]);
}

}  // namespace

struct Context::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> direct = nil;
  id<MTLComputePipelineState> tree = nil;
  id<MTLBuffer> bodies = nil;
  id<MTLBuffer> cells = nil;
  id<MTLBuffer> cell_list = nil;
  id<MTLBuffer> leaf_list = nil;
  id<MTLBuffer> groups = nil;
  id<MTLBuffer> out = nil;

  // Shared storage: on Apple Silicon the CPU and GPU see the same memory, so
  // "uploading" is a memcpy into the buffer's contents.
  void ensure(id<MTLBuffer> __strong &buf, size_t bytes) {
    bytes = std::max<size_t>(bytes, 16);
    if (buf == nil || [buf length] < bytes) {
      buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      if (buf == nil) throw std::runtime_error("Metal: failed to allocate a buffer");
    }
  }
  void upload(id<MTLBuffer> __strong &buf, const void *data, size_t bytes) {
    ensure(buf, bytes);
    if (bytes) std::memcpy([buf contents], data, bytes);
  }
  // Commit, wait, and turn a GPU-side failure into a message.
  std::string finish(id<MTLCommandBuffer> cb, const char *what) {
    [cb commit];
    [cb waitUntilCompleted];
    if ([cb status] == MTLCommandBufferStatusError)
      return std::string("Metal: ") + what + " failed: " + describe([cb error]);
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

void Context::direct(const std::vector<kern::float4> &bodies, const kern::GTParams &p,
                     std::vector<kern::float4> &out) {
  const size_t n = bodies.size();
  out.resize(n);
  if (n == 0) return;
  std::string error;
  @autoreleasepool {
   try {
    Impl &m = *impl_;
    m.upload(m.bodies, bodies.data(), n * sizeof(kern::float4));
    m.ensure(m.out, n * sizeof(kern::float4));
    // The kernel's tile holds 256 bodies, so threadgroups must not be larger.
    const NSUInteger tg = std::min<NSUInteger>(256, [m.direct maxTotalThreadsPerThreadgroup]);
    const NSUInteger groups = (n + tg - 1) / tg;
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m.direct];
    [enc setBuffer:m.bodies offset:0 atIndex:0];
    [enc setBuffer:m.out offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    error = m.finish(cb, "direct kernel");
    if (error.empty()) std::memcpy(out.data(), [m.out contents], n * sizeof(kern::float4));
   } catch (const std::exception &e) {
    error = e.what();  // rethrown outside the autorelease pool
   }
  }
  if (!error.empty()) throw std::runtime_error(error);
}

void Context::tree(const std::vector<kern::float4> &bodies, const std::vector<kern::GTCell> &cells,
                   const std::vector<kern::uint> &cell_list,
                   const std::vector<kern::GTRange> &leaf_list,
                   const std::vector<kern::GTGroup> &groups, const kern::GTParams &p,
                   unsigned threads_per_group, std::vector<kern::float4> &out) {
  const size_t n = bodies.size();
  out.resize(n);
  if (n == 0 || groups.empty()) return;
  if (p.n != n || p.n_groups != groups.size())
    throw std::invalid_argument("Metal tree: parameters do not match the arrays");
  std::string error;
  @autoreleasepool {
   try {
    Impl &m = *impl_;
    if (threads_per_group == 0 || threads_per_group > [m.tree maxTotalThreadsPerThreadgroup]) {
      error = "Metal tree: " + std::to_string(threads_per_group) +
              " threads per group is more than this GPU allows (" +
              std::to_string((unsigned long)[m.tree maxTotalThreadsPerThreadgroup]) +
              "); use a smaller group_size";
    } else {
      m.upload(m.bodies, bodies.data(), n * sizeof(kern::float4));
      m.upload(m.cells, cells.data(), cells.size() * sizeof(kern::GTCell));
      m.upload(m.cell_list, cell_list.data(), cell_list.size() * sizeof(kern::uint));
      m.upload(m.leaf_list, leaf_list.data(), leaf_list.size() * sizeof(kern::GTRange));
      m.upload(m.groups, groups.data(), groups.size() * sizeof(kern::GTGroup));
      m.ensure(m.out, n * sizeof(kern::float4));
      id<MTLCommandBuffer> cb = [m.queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:m.tree];
      [enc setBuffer:m.bodies offset:0 atIndex:0];
      [enc setBuffer:m.cells offset:0 atIndex:1];
      [enc setBuffer:m.cell_list offset:0 atIndex:2];
      [enc setBuffer:m.leaf_list offset:0 atIndex:3];
      [enc setBuffer:m.groups offset:0 atIndex:4];
      [enc setBuffer:m.out offset:0 atIndex:5];
      [enc setBytes:&p length:sizeof(p) atIndex:6];
      [enc dispatchThreadgroups:MTLSizeMake(groups.size(), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
      [enc endEncoding];
      error = m.finish(cb, "tree kernel");
      if (error.empty()) std::memcpy(out.data(), [m.out contents], n * sizeof(kern::float4));
    }
   } catch (const std::exception &e) {
    error = e.what();  // rethrown outside the autorelease pool
   }
  }
  if (!error.empty()) throw std::runtime_error(error);
}

}  // namespace metal
}  // namespace galactic_tides
