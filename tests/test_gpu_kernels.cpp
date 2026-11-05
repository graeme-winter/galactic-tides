// Static and emulated checks of the GPU kernels, runnable on any platform.
//
// The arithmetic in include/galactic_tides/kernels/gravity_kernels.h is compiled
// here as C++ and compared against the double precision reference. The
// shared code and the Metal wrappers are also linted for constructs Metal
// rejects, so most mistakes show up without a Mac.

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <random>
#include <regex>
#include <string>
#include <vector>

#include "check.h"
#include "galactic_tides/gravity.h"
#include "galactic_tides/gravity_backends.h"
#include "galactic_tides/group_lists.h"
#include "galactic_tides/kernels/gravity_kernels.h"
#include "galactic_tides/metal_backend.h"

using namespace galactic_tides;

namespace {

// Same lopsided test system as test_octree: a Hernquist sphere with 10% of
// the mass in two compact satellites.
Particles make_test_system(std::size_t n, unsigned seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  Particles p;
  for (std::size_t i = 0; i < n; ++i) {
    Vec3 centre;
    double a = 1.0;
    if (i % 10 == 0) {
      centre = (i % 20 == 0) ? Vec3(4, 0, 0) : Vec3(-1, 3, 2);
      a = 0.1;
    }
    double x = std::sqrt(u(rng) * 0.95);
    double r = a * x / (1 - x);
    double cz = 2 * u(rng) - 1, ph = 2 * M_PI * u(rng), s = std::sqrt(1 - cz * cz);
    p.push_back(centre + r * Vec3(s * std::cos(ph), s * std::sin(ph), cz), Vec3(), 1.0 / n, i,
                ParticleType::Halo);
  }
  return p;
}

std::string strip_comments(const std::string &src) {
  std::string out;
  for (std::size_t i = 0; i < src.size();) {
    if (src.compare(i, 2, "//") == 0) {
      while (i < src.size() && src[i] != '\n') ++i;
    } else if (src.compare(i, 2, "/*") == 0) {
      auto e = src.find("*/", i + 2);
      i = e == std::string::npos ? src.size() : e + 2;
    } else {
      out += src[i++];
    }
  }
  return out;
}

// What Metal compiles: the embedded source without comments and without the
// C++-only branch of the platform block (#ifdef __METAL_VERSION__ ... #else
// ... #endif).
std::string metal_view(const std::string &src) {
  std::string s = strip_comments(src);
  const auto block = s.find("#ifdef __METAL_VERSION__");
  const auto b = s.find("#else", block);
  const auto e = s.find("#endif", b);
  if (block != std::string::npos && b != std::string::npos && e != std::string::npos)
    s.erase(b, e - b);
  return s;
}

int lint(const std::string &code, const char *what, bool quiet = false) {
  int problems = 0;
  struct Rule {
    const char *pattern, *message;
  } rules[] = {
      {R"((^|[^\w.])\d+\.\d*([eE][-+]?\d+)?(?![fFuU\w]))", "floating literal without an f suffix"},
      {R"((^|[^\w.])\.\d+([eE][-+]?\d+)?(?![fF\w]))", "floating literal without an f suffix"},
      {R"((^|[^\w.])\d+[eE][-+]?\d+(?![fF\w]))", "floating literal without an f suffix"},
      {R"(\bdouble\b)", "double is not available on Apple GPUs"},
      {R"(\bstd::)", "no standard library in Metal"},
      {R"(\b(new|delete|throw|try|virtual)\b)", "not allowed in Metal"},
      {R"(\b(float|uint|GTParams|GTAcc|float4)\s*&)", "references need an address space in Metal"},
  };
  for (const auto &r : rules) {
    std::regex re(r.pattern);
    for (auto it = std::sregex_iterator(code.begin(), code.end(), re); it != std::sregex_iterator();
         ++it) {
      // Allow "constant GTParams &" in kernel signatures: that is Metal syntax.
      const std::size_t pos = static_cast<std::size_t>(it->position());
      if (std::string(r.message).find("references") != std::string::npos &&
          code.rfind("constant", pos) != std::string::npos &&
          pos - code.rfind("constant", pos) < 12)
        continue;
      std::size_t line = std::count(code.begin(), code.begin() + it->position(), '\n') + 1;
      if (!quiet) std::cerr << what << " (comment-stripped) line " << line << ": " << r.message << ": '"
                << it->str() << "'\n";
      ++problems;
    }
  }
  return problems;
}

// Every body must be covered exactly once by each group's list (cells count
// all the bodies beneath them), and no accepted cell may contain one of the
// group's own targets. Returns the number of violations.
int check_lists(const Octree &tree, const GroupLists &L, std::size_t n, int group_size) {
  int bad = 0;
  // Groups tile [0, n) in order, each no bigger than group_size.
  std::uint32_t next = 0;
  for (const auto &g : L.groups) {
    if (g.begin != next || g.end <= g.begin || g.end - g.begin > std::uint32_t(group_size)) ++bad;
    next = g.end;
  }
  if (next != n) ++bad;
  std::vector<int> cover(n);
  for (const auto &g : L.groups) {
    std::fill(cover.begin(), cover.end(), 0);
    for (auto k = g.cell_begin; k < g.cell_end; ++k) {
      const auto &nd = tree.nodes()[L.cell_list[k]];
      if (nd.begin < g.end && g.begin < nd.end) ++bad;  // cell contains a target
      for (auto j = nd.begin; j < nd.end; ++j) ++cover[j];
    }
    for (auto k = g.body_begin; k < g.body_end; ++k) ++cover[L.body_list[k]];
    for (std::size_t j = 0; j < n; ++j) bad += cover[j] != 1;
  }
  return bad;
}

std::vector<double> sorted_errors(const Particles &a, const Particles &ref) {
  std::vector<double> e;
  for (std::size_t i = 0; i < a.size(); ++i) e.push_back(norm(a.acc[i] - ref.acc[i]) / norm(ref.acc[i]));
  std::sort(e.begin(), e.end());
  return e;
}

}  // namespace

int main() {
  // ---- layout: the host structs must match Metal's ----------------------------
  static_assert(sizeof(kern::float4) == 16 && alignof(kern::float4) == 16, "float4");
  static_assert(sizeof(kern::GTParams) == 32, "GTParams");
  static_assert(sizeof(kern::GTAcc) == 16, "GTAcc");
  static_assert(sizeof(kern::GTCell) == 48 && alignof(kern::GTCell) == 16, "GTCell");
  static_assert(sizeof(kern::GTGroup) == 32, "GTGroup");

  // ---- the lint itself catches what it should ---------------------------------
  CHECK(lint("float x = 2.5;", "", true) == 1);
  CHECK(lint("float x = .5;", "", true) == 1);
  CHECK(lint("float x = 1e-4;", "", true) == 1);
  CHECK(lint("float x = 2.5f * 1e-4f + .5f; uint k = 256u; a[3] = 0;", "", true) == 0);
  CHECK(lint("double y;", "", true) == 1);
  CHECK(lint("float r = std::sqrt(x);", "", true) == 1);
  CHECK(lint("void f(float &a);", "", true) == 1);
  CHECK(lint("kernel void k(constant GTParams &p [[buffer(2)]]);", "", true) == 0);

  // ---- the embedded Metal source -------------------------------------------
  const std::string src = metal::kKernelSource;
  CHECK(src.find("#ifndef GT_GRAVITY_KERNELS_H") != std::string::npos);
  // The shared header comes first and defines its guard, so the .metal
  // file's own #include of it is skipped when compiled from this string.
  const auto guard_def = src.find("#define GT_GRAVITY_KERNELS_H");
  const auto metal_include = src.find("#include \"gravity_kernels.h\"");
  CHECK(guard_def != std::string::npos && metal_include != std::string::npos &&
        guard_def < metal_include);
  CHECK(src.find("#include <metal_stdlib>") < src.find("kernel void"));
  for (const char *name : {"gt_direct", "gt_tree"})
    CHECK(src.find(std::string("kernel void ") + name + "(") != std::string::npos);
  CHECK(lint(metal_view(src), "kernel source") == 0);

  // ---- emulated direct kernel against the double precision reference -------
  const std::size_t n = 3000;
  Particles ref = make_test_system(n, 11);
  GravityParams gp;
  gp.eps = 0.01;
  compute_gravity_direct(ref, gp);

  gpu_vector<kern::float4> bodies;
  pack_bodies(ref.pos, ref.mass, Vec3(1.0, 1.0, 1.0), bodies);
  const kern::GTParams kp = make_kernel_params(gp, n);
  std::vector<double> err_a, err_p;
  for (kern::uint i = 0; i < n; ++i) {
    const kern::GTAcc a = kern::gt_direct_accel(i, bodies.data(), kp);
    const Vec3 acc(a.ax, a.ay, a.az);
    err_a.push_back(norm(acc - ref.acc[i]) / norm(ref.acc[i]));
    err_p.push_back(std::fabs(a.pot / ref.pot[i] - 1));
  }
  std::sort(err_a.begin(), err_a.end());
  std::sort(err_p.begin(), err_p.end());
  std::cout << "direct kernel (float) vs double: median " << err_a[n / 2] << " max "
            << err_a.back() << ", potential max " << err_p.back() << "\n";
  CHECK(err_a[n / 2] < 1e-5);
  CHECK(err_a.back() < 1e-3);
  CHECK(err_p.back() < 1e-5);

  // A body never feels itself, and a lone body feels nothing.
  {
    gpu_vector<kern::float4> one = {kern::float4{0.5f, 0.0f, 0.0f, 1.0f}};
    kern::GTParams p1 = make_kernel_params(gp, 1);
    const kern::GTAcc a = kern::gt_direct_accel(0, one.data(), p1);
    CHECK(a.ax == 0.0f && a.pot == 0.0f);
  }

  // DirectGravity on the CPU is the double reference.
  {
    Particles p = ref;
    DirectGravity d(gp, Device::CPU);
    d.compute(p);
    CHECK(norm(p.acc[17] - ref.acc[17]) == 0.0);
    CHECK(d.stats().particle_interactions == n * (n - 1));
  }

  // ---- grouped tree: list bookkeeping ----------------------------------------
  for (bool pack : {false, true})
  for (int gs : {1, 7, 32, 64}) {
    Octree tree;
    tree.build(ref.pos, ref.mass, 8);
    GroupLists L;
    GravityParams g2 = gp;
    g2.theta = 0.7;
    build_group_lists(tree, g2, gs, L, pack);
    const int bad = check_lists(tree, L, n, gs);
    std::cout << (pack ? "packed " : "") << "group_size " << gs << ": " << L.groups.size()
              << " groups, largest "
              << L.max_group << ", interactions/particle "
              << double(L.cell_interactions + L.body_interactions) / n << ", violations " << bad
              << "\n";
    CHECK(bad == 0);
    CHECK(L.max_group <= gs);
  }
  // Coincident particles make a leaf bigger than a group: it must be split.
  {
    Particles c;
    for (int i = 0; i < 100; ++i) c.push_back(Vec3(1, 2, 3), Vec3(), 1.0, i, ParticleType::Halo);
    for (int i = 0; i < 50; ++i) c.push_back(Vec3(i, -i, 0.5 * i), Vec3(), 1.0, 100 + i, ParticleType::Halo);
    Octree tree;
    tree.build(c.pos, c.mass, 8);
    for (bool pack : {false, true}) {
      GroupLists L;
      build_group_lists(tree, gp, 32, L, pack);
      CHECK(check_lists(tree, L, c.size(), 32) == 0);
      CHECK(L.max_group <= 32);
    }
  }

  // ---- arrays the GPU wraps in place must be page aligned -----------------------
  {
    auto aligned = [](const void *ptr) {
      return reinterpret_cast<std::uintptr_t>(ptr) % page_size() == 0;
    };
    gpu_vector<kern::float4> v(1001);
    CHECK(aligned(v.data()));
    CHECK(allocation_bytes(v) % page_size() == 0 && allocation_bytes(v) >= v.capacity() * 16);
    Octree tree;
    tree.build(ref.pos, ref.mass, 8);
    GroupLists L;
    build_group_lists(tree, gp, 32, L, true);
    CHECK(aligned(L.bodies.data()) && aligned(L.cells.data()) && aligned(L.cell_list.data()) &&
          aligned(L.body_list.data()) && aligned(L.groups.data()));
    // Rebuilding with the same particles keeps the storage, so the GPU's
    // wrappers can be reused from step to step.
    const void *before = L.cell_list.data();
    build_group_lists(tree, gp, 32, L, true);
    CHECK(L.cell_list.data() == before);
  }

  // ---- grouped tree: accuracy -------------------------------------------------
  // The group criterion is never looser than the per-particle one, so errors
  // should be no worse than the CPU tree's at the same opening angle (plus
  // single precision rounding).
  for (double theta : {0.5, 0.7, 1.0}) {
    GravityParams g2 = gp;
    g2.theta = theta;
    Particles t = ref, q = ref;
    Octree tree;
    compute_gravity_tree(t, g2, tree);
    GroupTreeGravity grp(g2, Device::CPU, 32, 8, theta == 0.7 ? 1 : 0);  // packed once
    grp.compute(q);
    const auto et = sorted_errors(t, ref), eq = sorted_errors(q, ref);
    std::cout << "theta " << theta << ": cpu tree median " << et[n / 2] << " p99 "
              << et[n * 99 / 100] << " | grouped median " << eq[n / 2] << " p99 "
              << eq[n * 99 / 100] << "\n";
    CHECK(eq[n / 2] <= 1.05 * et[n / 2] + 2e-6);
    CHECK(eq[n * 99 / 100] <= 1.05 * et[n * 99 / 100] + 2e-5);
    // Potentials too.
    double worst = 0;
    for (std::size_t i = 0; i < n; ++i) worst = std::max(worst, std::fabs(q.pot[i] / ref.pot[i] - 1));
    CHECK(worst < 2e-3);
  }

  // GroupTreeGravity scatters tree-order results back to the right particles.
  {
    GravityParams g2 = gp;
    g2.theta = 0.6;
    Particles q = ref;
    GroupTreeGravity grp(g2, Device::CPU, 32, 8);
    grp.compute(q);
    kern::GTParams kp = make_kernel_params(g2, n);
    kp.n_groups = static_cast<kern::uint>(grp.lists().groups.size());
    gpu_vector<kern::float4> out;
    evaluate_group_lists_cpu(grp.lists(), kp, out);
    const auto &order = grp.tree().order();
    int mismatches = 0;
    for (std::size_t k = 0; k < n; ++k)
      mismatches += q.acc[order[k]].x != double(out[k].x) || q.pot[order[k]] != double(out[k].w);
    CHECK(mismatches == 0);
    CHECK(grp.stats().cell_interactions > 0 && grp.stats().particle_interactions > 0);
  }

  // Without Metal the backend must refuse clearly rather than misbehave.
  if (!metal::compiled_in()) {
    bool threw = false;
    try {
      DirectGravity d(gp, Device::Metal);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
      GroupTreeGravity g(gp, Device::Metal);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    CHECK(threw);
  }
  return TEST_MAIN_RESULT();
}
