#pragma once

#include <map>
#include <string>

#include "galactic_tides/particles.h"

namespace galactic_tides {

// GADGET-style HDF5 snapshots: /Header attributes and one /PartTypeN group per
// particle type with Coordinates, Velocities, ParticleIDs, Masses (and
// Potential when written by us). Readable by pynbody, yt, and friends; ICs from
// other generators in this format can be read back in.
//
// `metadata` is stored as string attributes on a /Parameters group so every
// snapshot records the configuration that produced it.
void write_snapshot(const std::string &path, const Particles &p, double time,
                    const std::map<std::string, std::string> &metadata = {});

// Reads all PartType groups present. Masses fall back to the Header MassTable
// when there is no Masses dataset; IDs are generated if absent. Particles are
// returned grouped by type in ascending type order.
Particles read_snapshot(const std::string &path, double *time = nullptr);

}  // namespace galactic_tides
