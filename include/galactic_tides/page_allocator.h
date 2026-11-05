#pragma once

// Page-aligned storage for arrays shared with the GPU. On Apple Silicon the
// CPU and GPU share memory, and Metal can wrap page-aligned host memory as a
// buffer without copying (newBufferWithBytesNoCopy). Arrays built on the CPU
// in these vectors are read and written by the GPU in place.

#include <cstddef>
#include <cstdlib>
#include <new>
#include <utility>
#include <vector>

#include <unistd.h>

namespace galactic_tides {

inline std::size_t page_size() {
  static const std::size_t size = [] {
    const long s = sysconf(_SC_PAGESIZE);
    return s > 0 ? static_cast<std::size_t>(s) : std::size_t{4096};
  }();
  return size;
}

// Whole pages covering `bytes` (at least one page).
inline std::size_t round_to_pages(std::size_t bytes) {
  const std::size_t p = page_size();
  return bytes == 0 ? p : (bytes + p - 1) / p * p;
}

template <class T>
struct PageAlignedAllocator {
  using value_type = T;
  PageAlignedAllocator() noexcept = default;
  template <class U>
  PageAlignedAllocator(const PageAlignedAllocator<U> &) noexcept {}

  T *allocate(std::size_t n) {
    if (n > static_cast<std::size_t>(-1) / sizeof(T) - page_size()) throw std::bad_array_new_length();
    void *p = nullptr;
    if (posix_memalign(&p, page_size(), round_to_pages(n * sizeof(T))) != 0) throw std::bad_alloc();
    return static_cast<T *>(p);
  }
  void deallocate(T *p, std::size_t) noexcept { std::free(p); }

  // resize() default-initialises rather than zero-fills: these arrays are
  // always written in full before they are read, and zeroing tens of MB
  // every step costs time.
  template <class U>
  void construct(U *p) noexcept {
    ::new (static_cast<void *>(p)) U;
  }
  template <class U, class... Args>
  void construct(U *p, Args &&...args) {
    ::new (static_cast<void *>(p)) U(std::forward<Args>(args)...);
  }

  template <class U>
  bool operator==(const PageAlignedAllocator<U> &) const noexcept { return true; }
  template <class U>
  bool operator!=(const PageAlignedAllocator<U> &) const noexcept { return false; }
};

template <class T>
using gpu_vector = std::vector<T, PageAlignedAllocator<T>>;

// The whole-page allocation behind v's storage: what the GPU may wrap.
template <class T>
std::size_t allocation_bytes(const gpu_vector<T> &v) {
  return v.capacity() == 0 ? 0 : round_to_pages(v.capacity() * sizeof(T));
}

}  // namespace galactic_tides
