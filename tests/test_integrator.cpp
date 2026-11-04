// Leapfrog on problems with known answers: a circular binary (period and
// energy) and a Plummer sphere in equilibrium (energy drift, virial ratio).

#include <cmath>
#include <random>

#include "check.h"
#include "galactic_tides/integrator.h"

using namespace galactic_tides;

int main() {
  // Equal mass circular binary, separation 1, G = M_total = 1.
  {
    Particles p;
    // Force on each body G m m / d^2 = 1/4 balances m v^2 / r = v^2, so v = 1/2.
    const double v = 0.5;
    p.push_back(Vec3(0.5, 0, 0), Vec3(0, v, 0), 0.5, 0, ParticleType::Halo);
    p.push_back(Vec3(-0.5, 0, 0), Vec3(0, -v, 0), 0.5, 1, ParticleType::Halo);
    GravityParams gp;
    gp.eps = 0.0;
    TreeGravity grav(gp);
    LeapfrogKDK lf(grav);
    lf.prime(p);
    const auto d0 = compute_diagnostics(p, 0.0);
    const double period = 2 * M_PI;  // 2 pi sqrt(a^3 / G M) with a = M = 1
    const int steps = 2000;
    const double dt = 10 * period / steps;
    for (int s = 0; s < steps; ++s) lf.step(p, dt);
    const auto d1 = compute_diagnostics(p, 10 * period);
    CHECK_CLOSE(d1.total(), d0.total(), 1e-5 * std::fabs(d0.total()));
    // Back where it started after 10 orbits (leapfrog phase error is O(dt^2)).
    CHECK(norm(p.pos[0] - Vec3(0.5, 0, 0)) < 0.02);
    CHECK_CLOSE(d1.angular_momentum.z, d0.angular_momentum.z, 1e-12);
  }

  // Plummer sphere, isotropic, sampled from the distribution function
  // (Aarseth, Henon & Wielen 1974). Henon units: G = M = 1, E = -1/4.
  {
    const std::size_t n = 2000;
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> u(0, 1);
    auto dir = [&]() {
      double cz = 2 * u(rng) - 1, ph = 2 * M_PI * u(rng), s = std::sqrt(1 - cz * cz);
      return Vec3(s * std::cos(ph), s * std::sin(ph), cz);
    };
    const double a = 3 * M_PI / 16;
    Particles p;
    for (std::size_t i = 0; i < n; ++i) {
      double r;
      do r = a / std::sqrt(std::pow(u(rng), -2.0 / 3.0) - 1); while (r > 10 * a);
      double q, g;
      do {
        q = u(rng);
        g = 0.1 * u(rng);
      } while (g > q * q * std::pow(1 - q * q, 3.5));
      const double vesc = std::sqrt(2.0) * std::pow(r * r + a * a, -0.25);
      p.push_back(r * dir(), q * vesc * dir(), 1.0 / n, i, ParticleType::Halo);
    }
    GravityParams gp;
    gp.eps = 0.02;
    gp.theta = 0.6;
    TreeGravity grav(gp);
    LeapfrogKDK lf(grav);
    lf.prime(p);
    const auto d0 = compute_diagnostics(p, 0);
    std::cout << "plummer: E0 " << d0.total() << " virial " << d0.virial_ratio() << "\n";
    CHECK(std::fabs(d0.virial_ratio() - 1.0) < 0.1);
    double t = 0;
    const double dt = 0.01;
    for (int s = 0; s < 500; ++s, t += dt) lf.step(p, dt);
    const auto d1 = compute_diagnostics(p, t);
    const double de = std::fabs(d1.total() / d0.total() - 1);
    std::cout << "plummer: after t=" << t << " dE/E " << de << " virial " << d1.virial_ratio()
              << "\n";
    CHECK(de < 5e-3);
    CHECK(std::fabs(d1.virial_ratio() - 1.0) < 0.1);
  }
  return TEST_MAIN_RESULT();
}
