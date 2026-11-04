// Tree forces against direct summation, and neighbour queries against brute
// force, on a clumpy Hernquist sphere plus a few satellites (a lopsided mass
// distribution is where the Barnes 1994 criterion earns its keep).

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <vector>

#include "check.h"
#include "galactic_tides/gravity.h"

using namespace galactic_tides;

namespace {

Particles make_test_system(std::size_t n, unsigned seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  Particles p;
  p.reserve(n);
  auto random_dir = [&]() {
    double cz = 2 * u(rng) - 1, ph = 2 * M_PI * u(rng), s = std::sqrt(1 - cz * cz);
    return Vec3(s * std::cos(ph), s * std::sin(ph), cz);
  };
  for (std::size_t i = 0; i < n; ++i) {
    Vec3 centre;
    double a = 1.0;
    if (i % 10 == 0) {  // 10% of the mass in two compact satellites
      centre = (i % 20 == 0) ? Vec3(4, 0, 0) : Vec3(-1, 3, 2);
      a = 0.1;
    }
    // Hernquist sphere: M(<r)/M = r^2 / (r + a)^2, truncated.
    double x = std::sqrt(u(rng) * 0.95);
    double r = a * x / (1 - x);
    p.push_back(centre + r * random_dir(), Vec3(), 1.0 / n, i, ParticleType::Halo);
  }
  return p;
}

struct ErrorSummary {
  double median, p99;
};

ErrorSummary relative_errors(const Particles &tree, const Particles &direct) {
  std::vector<double> e;
  for (std::size_t i = 0; i < tree.size(); ++i)
    e.push_back(norm(tree.acc[i] - direct.acc[i]) / norm(direct.acc[i]));
  std::sort(e.begin(), e.end());
  return {e[e.size() / 2], e[static_cast<std::size_t>(0.99 * e.size())]};
}

}  // namespace

int main() {
  const std::size_t n = 4000;
  Particles direct = make_test_system(n, 42);
  GravityParams gp;
  gp.eps = 0.01;
  compute_gravity_direct(direct, gp);

  Octree tree;
  double prev_median = 0.0;
  for (double theta : {0.3, 0.5, 0.7, 1.0}) {
    for (bool quad : {false, true}) {
      Particles p = direct;
      gp.theta = theta;
      gp.quadrupole = quad;
      WalkStats s = compute_gravity_tree(p, gp, tree);
      auto err = relative_errors(p, direct);
      std::cout << "theta " << theta << (quad ? " quad" : " mono") << ": median "
                << err.median << " p99 " << err.p99 << " interactions/particle "
                << double(s.cell_interactions + s.particle_interactions) / n << "\n";
      // Loose bounds: these are regression guards, not a paper.
      CHECK(err.median < 0.02 * theta * theta);
      CHECK(err.p99 < 0.1 * theta);
      if (quad) CHECK(err.median < prev_median);  // quadrupole must help
      prev_median = err.median;

      if (quad && theta == 0.5) {
        double max_dphi = 0;
        for (std::size_t i = 0; i < n; ++i)
          max_dphi = std::max(max_dphi, std::fabs(p.pot[i] / direct.pot[i] - 1));
        CHECK(max_dphi < 1e-3);
      }
    }
  }

  // Momentum: sum of m a should be small compared with sum of m |a|.
  {
    Particles p = direct;
    gp.theta = 0.6;
    gp.quadrupole = true;
    compute_gravity_tree(p, gp, tree);
    Vec3 f;
    double fabs_sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
      f += p.mass[i] * p.acc[i];
      fabs_sum += p.mass[i] * norm(p.acc[i]);
    }
    CHECK(norm(f) / fabs_sum < 1e-3);
  }

  // Leaf size should not change the answer much.
  {
    Particles a = direct, b = direct;
    gp.theta = 0.5;
    compute_gravity_tree(a, gp, tree, 1);
    compute_gravity_tree(b, gp, tree, 16);
    CHECK(relative_errors(a, direct).median < 2e-3);
    CHECK(relative_errors(b, direct).median < 2e-3);
  }

  // Neighbour search.
  {
    tree.build(direct.pos, direct.mass);
    std::mt19937_64 rng(7);
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);
    for (int trial = 0; trial < 50; ++trial) {
      const Vec3 c = direct.pos[pick(rng)];
      const double r = 0.05 * (1 + trial % 10);
      std::set<std::size_t> got, want;
      tree.for_each_within(c, r, [&](std::size_t i, const Vec3 &, double) { got.insert(i); });
      for (std::size_t i = 0; i < n; ++i)
        if (norm2(direct.pos[i] - c) <= r * r) want.insert(i);
      CHECK(got == want);
    }
  }

  // Degenerate inputs: coincident particles and empty sets must not hang.
  {
    Particles p;
    for (int i = 0; i < 100; ++i) p.push_back(Vec3(1, 2, 3), Vec3(), 1.0, i, ParticleType::Halo);
    p.push_back(Vec3(0, 0, 0), Vec3(), 1.0, 100, ParticleType::Halo);
    gp.theta = 0.6;
    compute_gravity_tree(p, gp, tree);
    CHECK(std::isfinite(p.acc[0].x) && std::isfinite(p.acc[100].x));
    Particles empty;
    compute_gravity_tree(empty, gp, tree);
  }

  return TEST_MAIN_RESULT();
}
