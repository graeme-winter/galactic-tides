#include <cstdio>

#include "check.h"
#include "galactic_tides/snapshot_io.h"

using namespace galactic_tides;

int main() {
  Particles p;
  // Deliberately interleaved types: the writer groups them, the reader
  // returns them grouped in type order.
  p.push_back(Vec3(1, 2, 3), Vec3(4, 5, 6), 0.5, 10, ParticleType::Disk);
  p.push_back(Vec3(-1, 0, 1), Vec3(0, 1, 0), 2.0, 11, ParticleType::Halo);
  p.push_back(Vec3(7, 8, 9), Vec3(1, 1, 1), 0.25, 12, ParticleType::Disk);
  p.push_back(Vec3(0, 0, 0.5), Vec3(0, 0, -1), 0.1, 13, ParticleType::Bulge);
  p.pot = {-1.0, -2.0, -3.0, -4.0};

  const char *path = "test_snapshot_io.hdf5";
  write_snapshot(path, p, 12.5, {{"theta", "0.6"}, {"note", "test"}});
  double t = 0;
  Particles q = read_snapshot(path, &t);
  std::remove(path);

  CHECK_CLOSE(t, 12.5, 0);
  CHECK(q.size() == 4);
  CHECK(q.type[0] == ParticleType::Halo && q.id[0] == 11);
  CHECK(q.type[1] == ParticleType::Disk && q.id[1] == 10);
  CHECK(q.type[2] == ParticleType::Disk && q.id[2] == 12);
  CHECK(q.type[3] == ParticleType::Bulge && q.id[3] == 13);
  CHECK_CLOSE(q.pos[2].z, 9.0, 0);
  CHECK_CLOSE(q.vel[1].x, 4.0, 0);
  CHECK_CLOSE(q.mass[3], 0.1, 0);

  bool threw = false;
  try {
    read_snapshot("does_not_exist.hdf5");
  } catch (const std::exception &) {
    threw = true;
  }
  CHECK(threw);
  return TEST_MAIN_RESULT();
}
