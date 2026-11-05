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
  id<MTLBuffer> bodies = nil;
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
      }
    }
  }
  if (!error.empty()) throw std::runtime_error(error);
}

Context::~Context() = default;

std::string Context::device_name() const {
  @autoreleasepool {
    return std::string([[impl_->device name] UTF8String]);
  }
}

void Context::direct(const std::vector<kern::float4> &bodies, const kern::GTParams &p,
                     std::vector<kern::float4> &out) {
  const size_t n = bodies.size();
  out.resize(n);
  if (n == 0) return;
  std::string error;
  @autoreleasepool {
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
  }
  if (!error.empty()) throw std::runtime_error(error);
}

}  // namespace metal
}  // namespace galactic_tides
