// Structural and dynamical checks on the Hernquist (1993) realisation.

#include <algorithm>
#include <cmath>
#include <vector>

#include "check.h"
#include "galactic_tides/gravity.h"
#include "galactic_tides/hernquist_ics.h"

using namespace galactic_tides;

namespace {

double median(std::vector<double> v) {
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

// 2K / -sum(m r.a) for one particle type: 1 in virial equilibrium.
double virial(const Particles &p, ParticleType t) {
  double k2 = 0, w = 0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    if (p.type[i] != t) continue;
    k2 += p.mass[i] * norm2(p.vel[i]);
    w += p.mass[i] * dot(p.pos[i], p.acc[i]);
  }
  return -k2 / w;
}

}  // namespace

int main() {
  HernquistModel m;
  m.disk_n = 4000;
  m.bulge_n = 1500;
  m.halo_n = 8000;
  m.seed = 3;
  HernquistReport report;
  Particles p = make_hernquist_galaxy(m, &report, 100);

  // Bookkeeping.
  const auto counts = p.count_by_type();
  CHECK(counts[1] == 8000 && counts[2] == 4000 && counts[3] == 1500);
  double mt = 0;
  Vec3 cm, cv, lz_disk;
  for (std::size_t i = 0; i < p.size(); ++i) {
    mt += p.mass[i];
    cm += p.mass[i] * p.pos[i];
    cv += p.mass[i] * p.vel[i];
    if (p.type[i] == ParticleType::Disk) lz_disk += p.mass[i] * cross(p.pos[i], p.vel[i]);
  }
  CHECK_CLOSE(mt, m.total_mass(), 1e-9);
  CHECK(norm(cm) < 1e-12 && norm(cv) < 1e-12);
  CHECK(lz_disk.z > 0.5);  // ~ M_d * R * v: rotating anticlockwise
  CHECK(*std::min_element(p.id.begin(), p.id.end()) == 100);

  // Structure. Disk: mean R of an exponential is ~2h; sech^2 has median
  // |z| = z0 atanh(1/2). Bulge: half-mass radius of the truncated profile.
  std::vector<double> Rd, zd, rb;
  for (std::size_t i = 0; i < p.size(); ++i) {
    if (p.type[i] == ParticleType::Disk) {
      Rd.push_back(std::hypot(p.pos[i].x, p.pos[i].y));
      zd.push_back(std::fabs(p.pos[i].z));
    } else if (p.type[i] == ParticleType::Bulge) {
      rb.push_back(norm(p.pos[i]));
    }
  }
  double meanR = 0;
  for (double R : Rd) meanR += R / Rd.size();
  CHECK_CLOSE(meanR, 2.0 * m.disk_h, 0.1);
  CHECK_CLOSE(median(zd), m.disk_z0 * std::atanh(0.5), 0.01);
  {
    const double f = std::pow(m.bulge_rmax / (m.bulge_rmax + m.bulge_a), 2);
    const double x = std::sqrt(0.5 * f);
    CHECK_CLOSE(median(rb), m.bulge_a * x / (1 - x), 0.04);
  }

  // Toomre Q at the reference radius.
  CHECK(!report.R.empty());
  {
    std::vector<double> &R = report.R;
    auto it = std::lower_bound(R.begin(), R.end(), m.disk_Q_radius);
    const std::size_t j = static_cast<std::size_t>(it - R.begin());
    const double t = (m.disk_Q_radius - R[j - 1]) / (R[j] - R[j - 1]);
    const double Q = report.Q[j - 1] + t * (report.Q[j] - report.Q[j - 1]);
    CHECK_CLOSE(Q, m.disk_Q, 0.05);
  }
  // Flat-ish rotation curve of order unity in these units.
  for (std::size_t i = 0; i < report.R.size(); ++i)
    if (report.R[i] > 1.5 && report.R[i] < 8) CHECK(report.vc[i] > 0.7 && report.vc[i] < 1.1);

  // Dynamics. With the refinements on, every component should be close to
  // virial equilibrium in the measured forces; plain Hernquist (1993) leaves
  // the halo a few percent cold.
  GravityParams gp = m.gravity;
  Octree tree;
  compute_gravity_tree(p, gp, tree);
  const double halo_plain = virial(p, ParticleType::Halo);
  std::cout << "plain: halo " << halo_plain << " disk " << virial(p, ParticleType::Disk)
            << " bulge " << virial(p, ParticleType::Bulge) << "\n";
  CHECK(halo_plain > 0.85 && halo_plain < 0.97);

  m.preserve_dispersion = true;
  m.disk_sigma_z_jeans = true;
  Particles q = make_hernquist_galaxy(m);
  compute_gravity_tree(q, gp, tree);
  std::cout << "refined: halo " << virial(q, ParticleType::Halo) << " disk "
            << virial(q, ParticleType::Disk) << " bulge " << virial(q, ParticleType::Bulge)
            << "\n";
  CHECK_CLOSE(virial(q, ParticleType::Halo), 1.0, 0.04);
  CHECK_CLOSE(virial(q, ParticleType::Disk), 1.0, 0.06);
  CHECK_CLOSE(virial(q, ParticleType::Bulge), 1.0, 0.06);

  // Encounter orbit: parabolic means zero orbital energy.
  {
    Vec3 dx, dv;
    const double G = 1, m1 = 7, m2 = 3, rp = 2, r0 = 20;
    parabolic_orbit(G, m1, m2, rp, r0, dx, dv);
    CHECK_CLOSE(norm(dx), r0, 1e-12);
    CHECK_CLOSE(0.5 * norm2(dv) - G * (m1 + m2) / norm(dx), 0.0, 1e-12);
    CHECK_CLOSE(cross(dx, dv).z, std::sqrt(2 * G * (m1 + m2) * rp), 1e-12);
    CHECK(dot(dx, dv) < 0);  // approaching
  }

  // Orientation is a rotation: it preserves |L| and tilts the spin axis.
  {
    Particles r = p;
    orient(r, 90.0, 0.0);
    Vec3 L;
    for (std::size_t i = 0; i < r.size(); ++i)
      if (r.type[i] == ParticleType::Disk) L += r.mass[i] * cross(r.pos[i], r.vel[i]);
    CHECK_CLOSE(norm(L), norm(lz_disk), 1e-9);
    // 90 degrees about x maps (x, y, z) -> (x, -z, y).
    CHECK_CLOSE(L.y, -lz_disk.z, 1e-9);
    CHECK_CLOSE(L.z, lz_disk.y, 1e-9);
  }
  return TEST_MAIN_RESULT();
}
