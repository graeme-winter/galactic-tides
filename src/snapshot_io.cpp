#include "galactic_tides/snapshot_io.h"

#include <hdf5.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace galactic_tides {

namespace {

// RAII for hid_t with the matching close function.
class Handle {
 public:
  Handle(hid_t id, herr_t (*closer)(hid_t), const std::string &what) : id_(id), close_(closer) {
    if (id_ < 0) throw std::runtime_error("HDF5: failed to " + what);
  }
  ~Handle() {
    if (id_ >= 0) close_(id_);
  }
  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;
  operator hid_t() const { return id_; }

 private:
  hid_t id_;
  herr_t (*close_)(hid_t);
};

void check(herr_t status, const std::string &what) {
  if (status < 0) throw std::runtime_error("HDF5: failed to " + what);
}

template <class T>
hid_t native_type();
template <>
hid_t native_type<double>() { return H5T_NATIVE_DOUBLE; }
template <>
hid_t native_type<std::uint32_t>() { return H5T_NATIVE_UINT32; }
template <>
hid_t native_type<std::uint64_t>() { return H5T_NATIVE_UINT64; }
template <>
hid_t native_type<std::int32_t>() { return H5T_NATIVE_INT32; }

template <class T>
void write_attr(hid_t loc, const std::string &name, const T *data, hsize_t n) {
  Handle space(n == 1 ? H5Screate(H5S_SCALAR) : H5Screate_simple(1, &n, nullptr), H5Sclose,
               "create dataspace");
  Handle attr(H5Acreate2(loc, name.c_str(), native_type<T>(), space, H5P_DEFAULT, H5P_DEFAULT),
              H5Aclose, "create attribute " + name);
  check(H5Awrite(attr, native_type<T>(), data), "write attribute " + name);
}

void write_string_attr(hid_t loc, const std::string &name, const std::string &value) {
  Handle type(H5Tcopy(H5T_C_S1), H5Tclose, "copy string type");
  check(H5Tset_size(type, value.empty() ? 1 : value.size()), "set string size");
  Handle space(H5Screate(H5S_SCALAR), H5Sclose, "create dataspace");
  Handle attr(H5Acreate2(loc, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose,
              "create attribute " + name);
  check(H5Awrite(attr, type, value.empty() ? "" : value.c_str()), "write attribute " + name);
}

template <class T>
void write_dataset(hid_t loc, const std::string &name, const T *data, hsize_t rows,
                   hsize_t cols) {
  hsize_t dims[2] = {rows, cols};
  Handle space(H5Screate_simple(cols == 1 ? 1 : 2, dims, nullptr), H5Sclose,
               "create dataspace");
  Handle ds(H5Dcreate2(loc, name.c_str(), native_type<T>(), space, H5P_DEFAULT, H5P_DEFAULT,
                       H5P_DEFAULT),
            H5Dclose, "create dataset " + name);
  if (rows > 0)
    check(H5Dwrite(ds, native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
          "write dataset " + name);
}

bool exists(hid_t loc, const std::string &name) { return H5Lexists(loc, name.c_str(), H5P_DEFAULT) > 0; }

template <class T>
std::vector<T> read_dataset(hid_t loc, const std::string &name, hsize_t expect_cols,
                            hsize_t *rows_out) {
  Handle ds(H5Dopen2(loc, name.c_str(), H5P_DEFAULT), H5Dclose, "open dataset " + name);
  Handle space(H5Dget_space(ds), H5Sclose, "get dataspace of " + name);
  int rank = H5Sget_simple_extent_ndims(space);
  hsize_t dims[2] = {0, 1};
  if (rank < 1 || rank > 2) throw std::runtime_error("unexpected rank for " + name);
  H5Sget_simple_extent_dims(space, dims, nullptr);
  if (rank == 1) dims[1] = 1;
  if (dims[1] != expect_cols)
    throw std::runtime_error("dataset " + name + " has " + std::to_string(dims[1]) +
                             " columns, expected " + std::to_string(expect_cols));
  std::vector<T> out(dims[0] * dims[1]);
  if (!out.empty())
    check(H5Dread(ds, native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()),
          "read dataset " + name);
  *rows_out = dims[0];
  return out;
}

template <class T, std::size_t N>
std::array<T, N> read_attr_array(hid_t loc, const std::string &name) {
  std::array<T, N> out{};
  if (H5Aexists(loc, name.c_str()) <= 0) return out;
  Handle attr(H5Aopen(loc, name.c_str(), H5P_DEFAULT), H5Aclose, "open attribute " + name);
  check(H5Aread(attr, native_type<T>(), out.data()), "read attribute " + name);
  return out;
}

}  // namespace

void write_snapshot(const std::string &path, const Particles &p, double time,
                    const std::map<std::string, std::string> &metadata) {
  Handle file(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose,
              "create " + path);
  const auto counts = p.count_by_type();
  {
    Handle header(H5Gcreate2(file, "Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose,
                  "create Header");
    std::array<std::uint32_t, 6> n32{}, zeros{};
    for (int t = 0; t < kNumParticleTypes; ++t) {
      if (counts[t] > 0xffffffffu) throw std::runtime_error("too many particles for header");
      n32[t] = static_cast<std::uint32_t>(counts[t]);
    }
    std::array<double, 6> mass_table{};
    write_attr(header, "NumPart_ThisFile", n32.data(), 6);
    write_attr(header, "NumPart_Total", n32.data(), 6);
    write_attr(header, "NumPart_Total_HighWord", zeros.data(), 6);
    write_attr(header, "MassTable", mass_table.data(), 6);
    double redshift = 0.0, box = 0.0;
    write_attr(header, "Time", &time, 1);
    write_attr(header, "Redshift", &redshift, 1);
    write_attr(header, "BoxSize", &box, 1);
    std::int32_t one = 1, zero = 0;
    write_attr(header, "NumFilesPerSnapshot", &one, 1);
    write_attr(header, "Flag_DoublePrecision", &one, 1);
    for (const char *flag : {"Flag_Sfr", "Flag_Cooling", "Flag_Feedback", "Flag_StellarAge",
                             "Flag_Metals"})
      write_attr(header, flag, &zero, 1);
  }
  if (!metadata.empty()) {
    Handle pg(H5Gcreate2(file, "Parameters", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose,
              "create Parameters");
    for (const auto &kv : metadata) write_string_attr(pg, kv.first, kv.second);
  }

  for (int t = 0; t < kNumParticleTypes; ++t) {
    if (counts[t] == 0) continue;
    std::vector<double> xyz, vxyz, m, phi;
    std::vector<std::uint64_t> ids;
    xyz.reserve(3 * counts[t]);
    vxyz.reserve(3 * counts[t]);
    for (std::size_t i = 0; i < p.size(); ++i) {
      if (static_cast<int>(p.type[i]) != t) continue;
      for (int k = 0; k < 3; ++k) {
        xyz.push_back(p.pos[i][k]);
        vxyz.push_back(p.vel[i][k]);
      }
      m.push_back(p.mass[i]);
      phi.push_back(p.pot[i]);
      ids.push_back(p.id[i]);
    }
    const std::string gname = "PartType" + std::to_string(t);
    Handle g(H5Gcreate2(file, gname.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose,
             "create " + gname);
    const hsize_t n = counts[t];
    write_dataset(g, "Coordinates", xyz.data(), n, 3);
    write_dataset(g, "Velocities", vxyz.data(), n, 3);
    write_dataset(g, "Masses", m.data(), n, 1);
    write_dataset(g, "ParticleIDs", ids.data(), n, 1);
    write_dataset(g, "Potential", phi.data(), n, 1);
  }
}

Particles read_snapshot(const std::string &path, double *time) {
  // Keep HDF5 quiet about probing for optional things; we report errors ourselves.
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  Handle file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "open " + path);
  std::array<double, 6> mass_table{};
  if (exists(file, "Header")) {
    Handle header(H5Gopen2(file, "Header", H5P_DEFAULT), H5Gclose, "open Header");
    mass_table = read_attr_array<double, 6>(header, "MassTable");
    if (time) *time = read_attr_array<double, 1>(header, "Time")[0];
  } else if (time) {
    *time = 0.0;
  }

  Particles p;
  std::uint64_t next_id = 0;
  for (int t = 0; t < kNumParticleTypes; ++t) {
    const std::string gname = "PartType" + std::to_string(t);
    if (!exists(file, gname)) continue;
    Handle g(H5Gopen2(file, gname.c_str(), H5P_DEFAULT), H5Gclose, "open " + gname);
    hsize_t n = 0, nv = 0, nm = 0, ni = 0;
    auto xyz = read_dataset<double>(g, "Coordinates", 3, &n);
    auto vxyz = read_dataset<double>(g, "Velocities", 3, &nv);
    if (nv != n) throw std::runtime_error(gname + ": Velocities/Coordinates length mismatch");
    std::vector<double> m;
    if (exists(g, "Masses")) {
      m = read_dataset<double>(g, "Masses", 1, &nm);
      if (nm != n) throw std::runtime_error(gname + ": Masses length mismatch");
    } else {
      if (mass_table[t] <= 0)
        throw std::runtime_error(gname + ": no Masses dataset and no MassTable entry");
      m.assign(n, mass_table[t]);
    }
    std::vector<std::uint64_t> ids;
    if (exists(g, "ParticleIDs")) {
      ids = read_dataset<std::uint64_t>(g, "ParticleIDs", 1, &ni);
      if (ni != n) throw std::runtime_error(gname + ": ParticleIDs length mismatch");
    }
    for (hsize_t i = 0; i < n; ++i) {
      const std::uint64_t ident = ids.empty() ? next_id++ : ids[i];
      p.push_back(Vec3(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]),
                  Vec3(vxyz[3 * i], vxyz[3 * i + 1], vxyz[3 * i + 2]), m[i], ident,
                  static_cast<ParticleType>(t));
    }
  }
  return p;
}

}  // namespace galactic_tides
