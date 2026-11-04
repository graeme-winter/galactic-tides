// Generate initial conditions: a single Hernquist (1993) galaxy, or two of
// them on a parabolic encounter orbit.
//
//   galactic-tides-ics params.par [key=value ...]

#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "galactic_tides/hernquist_ics.h"
#include "galactic_tides/params.h"
#include "galactic_tides/snapshot_io.h"

using namespace galactic_tides;

namespace {

void write_report(const std::string &path, const HernquistReport &r) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path);
  out << "# disk midplane\n# R vc kappa Sigma sigma_R sigma_phi sigma_z vphi_mean Q\n";
  for (std::size_t i = 0; i < r.R.size(); ++i)
    out << r.R[i] << ' ' << r.vc[i] << ' ' << r.kappa[i] << ' ' << r.sigma[i] << ' '
        << r.sigma_R[i] << ' ' << r.sigma_phi[i] << ' ' << r.sigma_z[i] << ' ' << r.vphi_mean[i]
        << ' ' << r.Q[i] << '\n';
  out << "\n\n# spheroids\n# r M(<r) sigma_halo sigma_bulge\n";
  for (std::size_t i = 0; i < r.r.size(); ++i)
    out << r.r[i] << ' ' << r.mass_enclosed[i] << ' ' << r.sigma_halo[i] << ' '
        << r.sigma_bulge[i] << '\n';
}

void summarise(const HernquistModel &m, const HernquistReport &r) {
  std::printf("  disk %zu  bulge %zu  halo %zu  total mass %.4g\n", m.disk_n, m.bulge_n,
              m.halo_n, m.total_mass());
  for (double R : {1.0, 2.43, 5.0, 8.0}) {
    for (std::size_t i = 0; i + 1 < r.R.size(); ++i) {
      if (r.R[i] <= R && r.R[i + 1] > R) {
        std::printf("  R = %-5.3g vc %.3f  vphi %.3f  sigma_R %.3f  sigma_z %.3f  Q %.2f\n", R,
                    r.vc[i], r.vphi_mean[i], r.sigma_R[i], r.sigma_z[i], r.Q[i]);
        break;
      }
    }
  }
  std::printf("  escape speed rejections: %zu", r.vesc_rejections);
  if (m.preserve_dispersion) std::printf("  (dispersion saturated for %zu)", r.dispersion_saturated);
  std::printf("\n  options: preserve_dispersion %s, disk_sigma_z %s\n",
              m.preserve_dispersion ? "on" : "off", m.disk_sigma_z_jeans ? "jeans" : "sheet");
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      std::cerr << "usage: " << argv[0] << " params.par [key=value ...]\n";
      return 2;
    }
    Params params;
    params.load_file(argv[1]);
    auto rest = params.apply_overrides(std::vector<std::string>(argv + 2, argv + argc));
    if (!rest.empty()) throw std::runtime_error("unexpected argument: " + rest[0]);

    const std::string mode = params.get_string("mode", "galaxy");
    const std::string output = params.get_string("output", "ics.hdf5");
    const std::string report_path = params.get_string("report", "ics_report.txt");
    HernquistModel model = HernquistModel::from_params(params);

    Particles all;
    HernquistReport report;
    if (mode == "galaxy") {
      std::printf("building galaxy (seed %llu)\n", (unsigned long long)model.seed);
      all = make_hernquist_galaxy(model, &report);
      orient(all, params.get_double("incl", 0.0), params.get_double("omega", 0.0));
      summarise(model, report);
    } else if (mode == "encounter") {
      const double rp = params.get_double("encounter_rp", 3.0);
      const double r0 = params.get_double("encounter_r0", 25.0);
      std::printf("building galaxy 1 (seed %llu)\n", (unsigned long long)model.seed);
      Particles g1 = make_hernquist_galaxy(model, &report);
      summarise(model, report);
      HernquistModel model2 = model;
      model2.seed = model.seed + 1;
      std::printf("building galaxy 2 (seed %llu)\n", (unsigned long long)model2.seed);
      Particles g2 = make_hernquist_galaxy(model2, nullptr, g1.size());
      orient(g1, params.get_double("g1_incl", 0.0), params.get_double("g1_omega", 0.0));
      orient(g2, params.get_double("g2_incl", 0.0), params.get_double("g2_omega", 0.0));
      const double m1 = model.total_mass(), m2 = model2.total_mass();
      Vec3 dx, dv;
      parabolic_orbit(model.G, m1, m2, rp, r0, dx, dv);
      // Put the centre of mass at rest at the origin.
      shift(g1, -(m2 / (m1 + m2)) * dx, -(m2 / (m1 + m2)) * dv);
      shift(g2, (m1 / (m1 + m2)) * dx, (m1 / (m1 + m2)) * dv);
      append(all, g1);
      append(all, g2);
      std::printf("parabolic orbit: pericentre %.3g, start separation %.3g, relative speed %.3g\n",
                  rp, r0, norm(dv));
    } else {
      throw std::runtime_error("mode must be 'galaxy' or 'encounter', got '" + mode + "'");
    }

    for (const auto &k : params.unused())
      std::fprintf(stderr, "warning: parameter '%s' was not used\n", k.c_str());
    write_snapshot(output, all, 0.0, params.used());
    write_report(report_path, report);
    std::printf("wrote %zu particles to %s, model tables to %s\n", all.size(), output.c_str(),
                report_path.c_str());
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
