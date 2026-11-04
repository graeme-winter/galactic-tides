#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "galactic_tides/octree.h"
#include "galactic_tides/params.h"
#include "galactic_tides/particles.h"

namespace galactic_tides {

// Disk / bulge / halo galaxy after Hernquist (1993), ApJS 86, 389,
// "N-body realizations of compound galaxies".
//
// Units: G = 1, disk mass = 1, disk scale length h = 1 by default. Scaled to
// the Milky Way (h = 3.5 kpc, M_disk = 5.6e10 Msun) the velocity unit is
// ~262 km/s and the time unit ~13 Myr.
//
// Components:
//   disk   rho = M_d / (4 pi h^2 z0) exp(-R/h) sech^2(z/z0), truncated at R_max
//   bulge  Hernquist (1990): rho ∝ a / (r (r + a)^3), truncated at r_max
//   halo   rho ∝ exp(-r^2/r_c^2) / (r^2 + gamma^2), truncated at r_max
//
// Velocities are moments-based, as in the paper:
//   * bulge and halo: isotropic Gaussians whose dispersion solves the
//     spherical Jeans equation in the spherically averaged radial force of
//     all components, truncated at a fraction of the local escape speed;
//   * disk: sigma_z^2 = pi G Sigma z0 (isothermal sheet); sigma_R^2 ∝
//     exp(-R/h) normalised to Toomre Q at a reference radius; sigma_phi from
//     the epicyclic approximation; mean rotation from the asymmetric drift
//     equation.
// All forces used in the moment equations (radial force for the spheroids,
// rotation curve and epicyclic frequency for the disk) are measured with the
// softened tree on the realised particles and angle averaged, so they match
// what the simulation will actually feel.
//
// This is the "approximate" scheme: the Maxwellian velocity distributions are
// not exact equilibria (Kazantzidis, Magorrian & Moore 2004), so expect some
// relaxation over the first few dynamical times, especially in the centre.
//
// Two optional refinements, both off by default so the default model is the
// 1993 recipe:
//   * preserve_dispersion: truncating a Maxwellian at ~0.95 v_esc discards
//     its high speed tail (~10% of the kinetic energy when v_esc ~ 3 sigma),
//     leaving the spheroids cold. With this on, the Gaussian width is inflated
//     so the *truncated* distribution has the Jeans second moment.
//   * disk_sigma_z = jeans: replace the isothermal sheet sigma_z^2 =
//     pi G Sigma z0 (disk self-gravity only) with the vertical Jeans equation
//     in the full realised potential, sigma_z^2(R) = integral rho |a_z| dz /
//     rho(R, 0), so the halo and bulge's vertical pull is included.
struct HernquistModel {
  double G = 1.0;

  std::size_t disk_n = 20000;
  double disk_mass = 1.0;
  double disk_h = 1.0;
  double disk_z0 = 0.2;
  double disk_rmax = 10.0;
  double disk_Q = 1.5;
  double disk_Q_radius = 2.43;  // the solar radius, 8.5 kpc / 3.5 kpc

  std::size_t bulge_n = 5000;
  double bulge_mass = 1.0 / 3.0;
  double bulge_a = 0.2;
  double bulge_rmax = 4.0;

  std::size_t halo_n = 60000;
  double halo_mass = 5.8;
  double halo_core = 1.0;     // gamma
  double halo_cutoff = 10.0;  // r_c
  double halo_rmax = 30.0;

  double vesc_fraction = 0.95;  // reject spheroid velocities faster than this
  bool preserve_dispersion = false;
  bool disk_sigma_z_jeans = false;  // parameter: disk_sigma_z = sheet | jeans
  std::uint64_t seed = 12345;
  GravityParams gravity;  // used to measure the realised potential

  static HernquistModel from_params(const Params &p, const std::string &prefix = "");
  double total_mass() const { return disk_mass + bulge_mass + halo_mass; }
};

// Radial tables describing the realised model, for checking and plotting.
struct HernquistReport {
  // disk midplane, as a function of cylindrical R
  std::vector<double> R, vc, kappa, sigma, sigma_R, sigma_phi, sigma_z, vphi_mean, Q;
  // spheroids, as a function of spherical r
  std::vector<double> r, sigma_halo, sigma_bulge, mass_enclosed;
  std::size_t vesc_rejections = 0;
  std::size_t dispersion_saturated = 0;  // preserve_dispersion could not be met
};

// Builds the galaxy centred at the origin with zero net momentum, disk in the
// x-y plane rotating anticlockwise about +z. IDs start at id_offset.
Particles make_hernquist_galaxy(const HernquistModel &m, HernquistReport *report = nullptr,
                                std::uint64_t id_offset = 0);

// Rotate a galaxy's disk spin axis: inclination i about x, then argument
// omega about z (angles in degrees). Positions and velocities both rotate.
void orient(Particles &p, double incl_deg, double omega_deg);
void shift(Particles &p, const Vec3 &dx, const Vec3 &dv);
void append(Particles &dst, const Particles &src);

// Relative position and velocity of galaxy 2 with respect to galaxy 1 on a
// parabolic two-point-mass orbit with pericentre rp, starting (approaching)
// at separation r0, orbit in the x-y plane. This is the classic Toomre &
// Toomre / Barnes & Hernquist encounter set-up.
void parabolic_orbit(double G, double m1, double m2, double rp, double r0, Vec3 &dx, Vec3 &dv);

}  // namespace galactic_tides
