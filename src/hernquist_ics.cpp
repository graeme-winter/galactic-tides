#include "galactic_tides/hernquist_ics.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <stdexcept>

#include "galactic_tides/gravity.h"

namespace galactic_tides {

namespace {

// Linear interpolation in a monotonic table; clamps at the ends.
double interp(const std::vector<double> &x, const std::vector<double> &y, double xv) {
  if (xv <= x.front()) return y.front();
  if (xv >= x.back()) return y.back();
  auto it = std::upper_bound(x.begin(), x.end(), xv);
  const std::size_t j = static_cast<std::size_t>(it - x.begin());
  const double t = (xv - x[j - 1]) / (x[j] - x[j - 1]);
  return y[j - 1] + t * (y[j] - y[j - 1]);
}

std::vector<double> log_grid(double lo, double hi, std::size_t n) {
  std::vector<double> g(n);
  const double a = std::log(lo), b = std::log(hi);
  for (std::size_t i = 0; i < n; ++i) g[i] = std::exp(a + (b - a) * double(i) / double(n - 1));
  return g;
}

// A truncated spherical density profile normalised to a given mass, with a
// tabulated cumulative mass for sampling and for the Jeans equation.
class SphericalComponent {
 public:
  SphericalComponent(std::function<double(double)> shape, double rmax, double mass,
                     double rmin_hint)
      : shape_(std::move(shape)), rmax_(rmax), mass_(mass) {
    r_.push_back(0.0);
    for (double r : log_grid(rmin_hint, rmax, 4000)) r_.push_back(r);
    m_.assign(r_.size(), 0.0);
    auto f = [&](double r) { return r > 0 ? 4 * M_PI * r * r * shape_(r) : 0.0; };
    for (std::size_t i = 1; i < r_.size(); ++i)
      m_[i] = m_[i - 1] + 0.5 * (f(r_[i - 1]) + f(r_[i])) * (r_[i] - r_[i - 1]);
    norm_ = mass_ / m_.back();
    for (auto &m : m_) m *= norm_;
  }
  double rho(double r) const { return (r > rmax_ || r <= 0) ? 0.0 : norm_ * shape_(r); }
  double mass_within(double r) const { return r >= rmax_ ? mass_ : interp(r_, m_, r); }
  double sample_radius(double u) const { return interp(m_, r_, u * mass_); }
  double rmax() const { return rmax_; }
  double mass() const { return mass_; }

 private:
  std::function<double(double)> shape_;
  double rmax_, mass_, norm_ = 1.0;
  std::vector<double> r_, m_;
};

Vec3 random_direction(std::mt19937_64 &rng) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const double cz = 2 * u(rng) - 1, ph = 2 * M_PI * u(rng), s = std::sqrt(1 - cz * cz);
  return {s * std::cos(ph), s * std::sin(ph), cz};
}

// Fraction of an exponential disk's mass inside x = R/h.
double exp_disk_fraction(double x) { return 1.0 - (1.0 + x) * std::exp(-x); }

// <v^2> / sigma^2 for an isotropic Gaussian truncated at speed x sigma.
double truncated_maxwell_moment(double x) {
  const double e = std::exp(-0.5 * x * x);
  const double n = std::sqrt(M_PI / 2) * std::erf(x / std::sqrt(2.0)) - x * e;
  if (n <= 0) return 0.6 * x * x;  // tiny x: uniform in the ball
  return 3.0 - x * x * x * e / n;
}

// Width of a Gaussian which, truncated at vmax, has <v^2> = 3 sigma_jeans^2.
// The largest <v^2> any truncation can give is 3/5 vmax^2 (uniform in the
// ball); beyond that we return a very broad Gaussian and flag saturation.
double inflated_sigma(double sigma_jeans, double vmax, bool &saturated) {
  saturated = false;
  if (sigma_jeans <= 0 || vmax <= 0) return sigma_jeans;
  const double target = 3 * sigma_jeans * sigma_jeans;
  auto moment = [&](double s) { return s * s * truncated_maxwell_moment(vmax / s); };
  double hi = sigma_jeans;
  for (int k = 0; k < 60 && moment(hi) < target; ++k) {
    hi *= 2;
    if (hi > 100 * vmax) {
      saturated = true;
      return hi;
    }
  }
  double lo = sigma_jeans;  // moment(lo) <= target always
  for (int it = 0; it < 60; ++it) {
    const double mid = 0.5 * (lo + hi);
    (moment(mid) < target ? lo : hi) = mid;
  }
  return 0.5 * (lo + hi);
}

}  // namespace

HernquistModel HernquistModel::from_params(const Params &p, const std::string &prefix) {
  HernquistModel m;
  auto d = [&](const char *k, double def) { return p.get_double(prefix + k, def); };
  auto n = [&](const char *k, std::size_t def) {
    long v = p.get_int(prefix + k, static_cast<long>(def));
    if (v < 0) throw std::runtime_error(prefix + k + " must be non-negative");
    return static_cast<std::size_t>(v);
  };
  m.G = p.get_double("G", m.G);
  m.disk_n = n("disk_n", m.disk_n);
  m.disk_mass = d("disk_mass", m.disk_mass);
  m.disk_h = d("disk_h", m.disk_h);
  m.disk_z0 = d("disk_z0", m.disk_z0);
  m.disk_rmax = d("disk_rmax", m.disk_rmax);
  m.disk_Q = d("disk_Q", m.disk_Q);
  m.disk_Q_radius = d("disk_Q_radius", m.disk_Q_radius);
  m.bulge_n = n("bulge_n", m.bulge_n);
  m.bulge_mass = d("bulge_mass", m.bulge_mass);
  m.bulge_a = d("bulge_a", m.bulge_a);
  m.bulge_rmax = d("bulge_rmax", m.bulge_rmax);
  m.halo_n = n("halo_n", m.halo_n);
  m.halo_mass = d("halo_mass", m.halo_mass);
  m.halo_core = d("halo_core", m.halo_core);
  m.halo_cutoff = d("halo_cutoff", m.halo_cutoff);
  m.halo_rmax = d("halo_rmax", m.halo_rmax);
  m.vesc_fraction = d("vesc_fraction", m.vesc_fraction);
  m.preserve_dispersion = p.get_bool(prefix + "preserve_dispersion", m.preserve_dispersion);
  const std::string sz = p.get_string(prefix + "disk_sigma_z", m.disk_sigma_z_jeans ? "jeans" : "sheet");
  if (sz != "sheet" && sz != "jeans")
    throw std::runtime_error(prefix + "disk_sigma_z must be 'sheet' or 'jeans'");
  m.disk_sigma_z_jeans = sz == "jeans";
  m.seed = static_cast<std::uint64_t>(p.get_int(prefix + "seed", static_cast<long>(m.seed)));
  m.gravity.G = m.G;
  m.gravity.eps = p.get_double("eps", m.gravity.eps);
  m.gravity.theta = p.get_double("theta", m.gravity.theta);
  m.gravity.quadrupole = p.get_bool("quadrupole", m.gravity.quadrupole);
  return m;
}

Particles make_hernquist_galaxy(const HernquistModel &m, HernquistReport *report,
                                std::uint64_t id_offset) {
  if (m.disk_n == 0 && m.bulge_n == 0 && m.halo_n == 0)
    throw std::invalid_argument("make_hernquist_galaxy: no particles requested");
  std::mt19937_64 rng(m.seed);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::normal_distribution<double> gauss(0.0, 1.0);
  auto u_open = [&]() {  // (0, 1)
    double u;
    do u = uni(rng); while (u <= 0.0 || u >= 1.0);
    return u;
  };

  const bool have_disk = m.disk_n > 0 && m.disk_mass > 0;
  const bool have_bulge = m.bulge_n > 0 && m.bulge_mass > 0;
  const bool have_halo = m.halo_n > 0 && m.halo_mass > 0;
  const double h = m.disk_h;

  const double gamma = m.halo_core, rc = m.halo_cutoff, a = m.bulge_a;
  SphericalComponent halo(
      [=](double r) { return std::exp(-r * r / (rc * rc)) / (r * r + gamma * gamma); },
      m.halo_rmax, have_halo ? m.halo_mass : 0.0, 1e-4 * (gamma > 0 ? std::min(gamma, rc) : rc));
  SphericalComponent bulge([=](double r) { return a / (r * std::pow(r + a, 3)); }, m.bulge_rmax,
                           have_bulge ? m.bulge_mass : 0.0, 1e-4 * a);
  const double disk_xmax = m.disk_rmax / h;
  const double disk_norm = exp_disk_fraction(disk_xmax);
  const double sigma0 = have_disk ? m.disk_mass / (2 * M_PI * h * h * disk_norm) : 0.0;
  auto surface_density = [&](double R) { return R > m.disk_rmax ? 0.0 : sigma0 * std::exp(-R / h); };
  auto disk_mass_within = [&](double r) {  // spherical approximation: thin disk
    if (!have_disk) return 0.0;
    return m.disk_mass * exp_disk_fraction(std::min(r, m.disk_rmax) / h) / disk_norm;
  };

  // ---- positions -------------------------------------------------------
  Particles p;
  p.reserve(m.disk_n + m.bulge_n + m.halo_n);
  std::uint64_t next_id = id_offset;
  if (have_halo) {
    for (std::size_t i = 0; i < m.halo_n; ++i)
      p.push_back(halo.sample_radius(u_open()) * random_direction(rng), Vec3(),
                  m.halo_mass / m.halo_n, next_id++, ParticleType::Halo);
  }
  if (have_disk) {
    for (std::size_t i = 0; i < m.disk_n; ++i) {
      // Invert the cumulative exponential profile by bisection.
      const double target = u_open() * disk_norm;
      double lo = 0.0, hi = disk_xmax;
      for (int it = 0; it < 60; ++it) {
        const double mid = 0.5 * (lo + hi);
        (exp_disk_fraction(mid) < target ? lo : hi) = mid;
      }
      const double R = 0.5 * (lo + hi) * h;
      const double phi = 2 * M_PI * uni(rng);
      const double z = m.disk_z0 * std::atanh(2 * u_open() - 1);
      p.push_back(Vec3(R * std::cos(phi), R * std::sin(phi), z), Vec3(),
                  m.disk_mass / m.disk_n, next_id++, ParticleType::Disk);
    }
  }
  if (have_bulge) {
    for (std::size_t i = 0; i < m.bulge_n; ++i)
      p.push_back(bulge.sample_radius(u_open()) * random_direction(rng), Vec3(),
                  m.bulge_mass / m.bulge_n, next_id++, ParticleType::Bulge);
  }

  // ---- the realised potential --------------------------------------------
  Octree tree;
  compute_gravity_tree(p, m.gravity, tree);

  // ---- spheroids: spherical isotropic Jeans equation -----------------------
  const double rlo = 1e-4 * std::min({gamma > 0 ? gamma : rc, a, h});
  const double rhi = std::max({m.halo_rmax, m.bulge_rmax, m.disk_rmax});
  const std::vector<double> rg = log_grid(rlo, rhi, 2000);
  std::vector<double> menc(rg.size()), gr(rg.size()), sig2_halo(rg.size()),
      sig2_bulge(rg.size());
  for (std::size_t i = 0; i < rg.size(); ++i)
    menc[i] = halo.mass_within(rg[i]) + bulge.mass_within(rg[i]) + disk_mass_within(rg[i]);
  {
    // Spherically averaged radial acceleration of the realised, softened
    // particle system, on a coarse log grid over a Fibonacci sphere of
    // directions. Using this rather than G M(<r) / r^2 keeps the spheroid
    // dispersions consistent with the softened forces they will feel.
    const std::size_t ng = 120, ndir = 64;
    const std::vector<double> rt = log_grid(10 * rlo, rhi, ng);
    std::vector<double> gt(ng);
    std::vector<Vec3> dirs(ndir);
    const double golden = M_PI * (3.0 - std::sqrt(5.0));
    for (std::size_t k = 0; k < ndir; ++k) {
      const double z = 1.0 - (2.0 * double(k) + 1.0) / double(ndir);
      const double s = std::sqrt(1.0 - z * z);
      dirs[k] = Vec3(s * std::cos(golden * double(k)), s * std::sin(golden * double(k)), z);
    }
#pragma omp parallel for schedule(dynamic)
    for (long i = 0; i < static_cast<long>(ng); ++i) {
      double sum = 0.0;
      for (const Vec3 &d : dirs) {
        Vec3 acc;
        double pot;
        tree.accel_pot(rt[i] * d, Octree::kNoSelf, m.gravity, acc, pot);
        sum += -dot(acc, d);
      }
      gt[i] = std::max(0.0, sum / double(ndir));
    }
    for (std::size_t i = 0; i < rg.size(); ++i) {
      if (rg[i] < rt.front()) gr[i] = gt.front() * rg[i] / rt.front();  // harmonic core
      else if (rg[i] > rt.back()) gr[i] = m.G * menc[i] / (rg[i] * rg[i]);
      else gr[i] = interp(rt, gt, rg[i]);
    }
  }
  auto jeans = [&](const SphericalComponent &c, std::vector<double> &sig2) {
    // sigma^2(r) = (1 / rho(r)) * integral_r^rmax rho(r') g(r') dr'
    double integral = 0.0;
    auto f = [&](std::size_t i) { return c.rho(rg[i]) * gr[i]; };
    sig2.back() = 0.0;
    for (std::size_t i = rg.size() - 1; i-- > 0;) {
      integral += 0.5 * (f(i) + f(i + 1)) * (rg[i + 1] - rg[i]);
      const double rho = c.rho(rg[i]);
      sig2[i] = rho > 0 ? integral / rho : 0.0;
    }
  };
  if (have_halo) jeans(halo, sig2_halo);
  if (have_bulge) jeans(bulge, sig2_bulge);

  std::size_t rejections = 0, saturated_count = 0;
  auto spheroid_velocity = [&](std::size_t i, const std::vector<double> &sig2) {
    const double r = norm(p.pos[i]);
    double sigma = std::sqrt(std::max(0.0, interp(rg, sig2, std::max(r, rlo))));
    const double vmax = m.vesc_fraction * std::sqrt(std::max(0.0, -2.0 * p.pot[i]));
    if (m.preserve_dispersion) {
      bool sat = false;
      sigma = inflated_sigma(sigma, vmax, sat);
      saturated_count += sat;
    }
    Vec3 v;
    for (int tries = 0;; ++tries) {
      v = sigma * Vec3(gauss(rng), gauss(rng), gauss(rng));
      if (norm(v) <= vmax) break;
      ++rejections;
      if (tries > 1000) {  // pathological: sigma >> v_esc; keep direction, cap speed
        v *= vmax / norm(v);
        break;
      }
    }
    return v;
  };

  // ---- disk: rotation curve and epicyclic frequency from the tree ---------
  const std::size_t nR = 200, nphi = 64;
  std::vector<double> Rg(nR), F(nR), omega2(nR), kappa2(nR), sigz2(nR, 0.0);
  const double dR = m.disk_rmax / double(nR);
  for (std::size_t i = 0; i < nR; ++i) Rg[i] = (double(i) + 1.0) * dR;
  if (have_disk) {
#pragma omp parallel for schedule(dynamic)
    for (long i = 0; i < static_cast<long>(nR); ++i) {
      double sum = 0.0;
      for (std::size_t k = 0; k < nphi; ++k) {
        const double phi = 2 * M_PI * (double(k) + 0.5 * (i % 2)) / double(nphi);
        const Vec3 x(Rg[i] * std::cos(phi), Rg[i] * std::sin(phi), 0.0);
        Vec3 acc;
        double pot;
        tree.accel_pot(x, Octree::kNoSelf, m.gravity, acc, pot);
        sum += -(acc.x * std::cos(phi) + acc.y * std::sin(phi));
      }
      F[i] = sum / double(nphi);
    }
    if (m.disk_sigma_z_jeans) {
      // sigma_z^2(R) = (1 / rho(R,0)) integral_0^inf rho(R,z) |a_z(R,z)| dz with
      // rho ∝ sech^2(z/z0), a_z from the tree (all components), azimuthally
      // averaged. For an isolated isothermal sheet this gives pi G Sigma z0.
      const std::size_t nz = 32, nphi_z = 16;
      const double zmax = 6.0 * m.disk_z0, dz = zmax / double(nz);
#pragma omp parallel for schedule(dynamic)
      for (long i = 0; i < static_cast<long>(nR); ++i) {
        double integral = 0.0;
        for (std::size_t j = 0; j < nz; ++j) {
          const double z = (double(j) + 0.5) * dz;
          double az = 0.0;
          for (std::size_t k = 0; k < nphi_z; ++k) {
            const double phi = 2 * M_PI * (double(k) + 0.5) / double(nphi_z);
            Vec3 acc;
            double pot;
            tree.accel_pot(Vec3(Rg[i] * std::cos(phi), Rg[i] * std::sin(phi), z), Octree::kNoSelf,
                           m.gravity, acc, pot);
            az += -acc.z;
          }
          const double sech = 1.0 / std::cosh(z / m.disk_z0);
          integral += sech * sech * (az / double(nphi_z)) * dz;
        }
        sigz2[i] = std::max(0.0, integral);
      }
    }
    for (std::size_t i = 0; i < nR; ++i) {
      double dF;
      if (i == 0) dF = (F[1] - F[0]) / dR;
      else if (i == nR - 1) dF = (F[i] - F[i - 1]) / dR;
      else dF = (F[i + 1] - F[i - 1]) / (2 * dR);
      omega2[i] = std::max(F[i] / Rg[i], 1e-30);
      // kappa^2 = R dOmega^2/dR + 4 Omega^2 = dF/dR + 3 F/R; keep it between
      // the Keplerian (Omega) and solid-body (2 Omega) limits.
      kappa2[i] = std::clamp(dF + 3 * F[i] / Rg[i], omega2[i], 4 * omega2[i]);
    }
  }

  auto disk_moments = [&](double R, double &vc2, double &kap2, double &sR2, double &sP2,
                          double &sZ2, double &vbar2) {
    const double o2 = interp(Rg, omega2, R);
    kap2 = interp(Rg, kappa2, R);
    vc2 = o2 * R * R;
    const double sig = surface_density(R);
    sZ2 = m.disk_sigma_z_jeans ? (R > m.disk_rmax ? 0.0 : interp(Rg, sigz2, R))
                               : M_PI * m.G * sig * m.disk_z0;
    const double kQ = std::sqrt(interp(Rg, kappa2, m.disk_Q_radius));
    const double sRQ = m.disk_Q * 3.36 * m.G * surface_density(m.disk_Q_radius) / kQ;
    sR2 = sRQ * sRQ * std::exp(-(R - m.disk_Q_radius) / h);
    const double ratio = kap2 / (4 * o2);
    sP2 = sR2 * ratio;
    vbar2 = std::max(0.0, vc2 + sR2 * (1 - ratio - 2 * R / h));
  };

  // ---- assign velocities ----------------------------------------------------
  for (std::size_t i = 0; i < p.size(); ++i) {
    switch (p.type[i]) {
      case ParticleType::Halo: p.vel[i] = spheroid_velocity(i, sig2_halo); break;
      case ParticleType::Bulge: p.vel[i] = spheroid_velocity(i, sig2_bulge); break;
      case ParticleType::Disk: {
        const double R = std::hypot(p.pos[i].x, p.pos[i].y);
        double vc2, kap2, sR2, sP2, sZ2, vbar2;
        disk_moments(R, vc2, kap2, sR2, sP2, sZ2, vbar2);
        const double vR = std::sqrt(sR2) * gauss(rng);
        const double vphi = std::sqrt(vbar2) + std::sqrt(sP2) * gauss(rng);
        const double vz = std::sqrt(sZ2) * gauss(rng);
        const double c = R > 0 ? p.pos[i].x / R : 1.0, s = R > 0 ? p.pos[i].y / R : 0.0;
        p.vel[i] = Vec3(vR * c - vphi * s, vR * s + vphi * c, vz);
        break;
      }
      default: break;
    }
  }

  // ---- centre of mass frame -------------------------------------------------
  double mt = 0.0;
  Vec3 cm, cv;
  for (std::size_t i = 0; i < p.size(); ++i) {
    mt += p.mass[i];
    cm += p.mass[i] * p.pos[i];
    cv += p.mass[i] * p.vel[i];
  }
  shift(p, -cm / mt, -cv / mt);

  if (report) {
    *report = HernquistReport{};
    report->vesc_rejections = rejections;
    report->dispersion_saturated = saturated_count;
    if (have_disk) {
      for (std::size_t i = 0; i < nR; ++i) {
        const double R = Rg[i];
        double vc2, kap2, sR2, sP2, sZ2, vbar2;
        disk_moments(R, vc2, kap2, sR2, sP2, sZ2, vbar2);
        const double sig = surface_density(R);
        report->R.push_back(R);
        report->vc.push_back(std::sqrt(vc2));
        report->kappa.push_back(std::sqrt(kap2));
        report->sigma.push_back(sig);
        report->sigma_R.push_back(std::sqrt(sR2));
        report->sigma_phi.push_back(std::sqrt(sP2));
        report->sigma_z.push_back(std::sqrt(sZ2));
        report->vphi_mean.push_back(std::sqrt(vbar2));
        report->Q.push_back(sig > 0 ? std::sqrt(sR2) * std::sqrt(kap2) / (3.36 * m.G * sig) : 0);
      }
    }
    for (std::size_t i = 0; i < rg.size(); i += 20) {
      report->r.push_back(rg[i]);
      report->sigma_halo.push_back(have_halo ? std::sqrt(sig2_halo[i]) : 0.0);
      report->sigma_bulge.push_back(have_bulge ? std::sqrt(sig2_bulge[i]) : 0.0);
      report->mass_enclosed.push_back(menc[i]);
    }
  }
  return p;
}

void orient(Particles &p, double incl_deg, double omega_deg) {
  const double i = incl_deg * M_PI / 180.0, w = omega_deg * M_PI / 180.0;
  const double ci = std::cos(i), si = std::sin(i), cw = std::cos(w), sw = std::sin(w);
  auto rot = [&](const Vec3 &v) {
    const Vec3 a(v.x, ci * v.y - si * v.z, si * v.y + ci * v.z);  // about x
    return Vec3(cw * a.x - sw * a.y, sw * a.x + cw * a.y, a.z);    // about z
  };
  for (std::size_t k = 0; k < p.size(); ++k) {
    p.pos[k] = rot(p.pos[k]);
    p.vel[k] = rot(p.vel[k]);
  }
}

void shift(Particles &p, const Vec3 &dx, const Vec3 &dv) {
  for (std::size_t k = 0; k < p.size(); ++k) {
    p.pos[k] += dx;
    p.vel[k] += dv;
  }
}

void append(Particles &dst, const Particles &src) {
  dst.reserve(dst.size() + src.size());
  for (std::size_t k = 0; k < src.size(); ++k) {
    dst.push_back(src.pos[k], src.vel[k], src.mass[k], src.id[k], src.type[k]);
    dst.pot.back() = src.pot[k];
    dst.acc.back() = src.acc[k];
  }
}

void parabolic_orbit(double G, double m1, double m2, double rp, double r0, Vec3 &dx, Vec3 &dv) {
  if (r0 < rp) throw std::invalid_argument("parabolic_orbit: start separation below pericentre");
  const double M = m1 + m2;
  const double f = -std::acos(std::clamp(2 * rp / r0 - 1, -1.0, 1.0));  // approaching
  const double hmom = std::sqrt(2 * G * M * rp);
  const double vr = G * M / hmom * std::sin(f);
  const double vt = hmom / r0;
  const Vec3 rhat(std::cos(f), std::sin(f), 0), fhat(-std::sin(f), std::cos(f), 0);
  dx = r0 * rhat;
  dv = vr * rhat + vt * fhat;
}

}  // namespace galactic_tides
