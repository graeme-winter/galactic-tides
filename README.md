# galactic-tides

A small, readable Barnes–Hut tree code for collisionless galaxy simulations,
with Hernquist (1993) disk/bulge/halo initial conditions. Written for fun and
for poking at, and structured so that SPH and collision handling can be added
later without rearranging the furniture.

## Building

Needs a C++17 compiler, CMake ≥ 3.16 and the HDF5 C library; OpenMP is used
if found.

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build        # tree accuracy, integrator, I/O, ICs
```

On macOS with Homebrew Clang/libomp you may need
`-DOpenMP_ROOT=$(brew --prefix libomp)`.

## Quick start

```sh
cd build
./galactic-tides-ics ../examples/galaxy_ics.par      # -> galaxy_ics.hdf5
./galactic-tides-run ../examples/galaxy_run.par      # -> galaxy/snap_*.hdf5
python3 ../scripts/plot_snapshot.py image galaxy/snap_0000.hdf5 galaxy/snap_0050.hdf5
python3 ../scripts/plot_snapshot.py profiles galaxy/snap_0000.hdf5 galaxy/snap_0050.hdf5
```

Any parameter can be overridden on the command line, e.g.
`./galactic-tides-ics ../examples/galaxy_ics.par disk_n=50000 seed=7`. Unused
parameters are reported, which catches typos.

Every snapshot is a valid starting point, so restarting is
`./galactic-tides-run ../examples/galaxy_run.par ics=galaxy/snap_0042.hdf5`;
snapshot numbering carries on from the snapshot's time.

For an encounter between two galaxies on a parabolic orbit:

```sh
./galactic-tides-ics ../examples/encounter_ics.par
./galactic-tides-run ../examples/encounter_run.par
python3 ../scripts/plot_snapshot.py frames encounter/snap_*.hdf5 --extent 40 --vmax 0.05
```

## Gravity solvers and the GPU

`galactic-tides-run` takes `gravity` and `device` parameters:

| gravity | device | what it is |
|---------|--------|------------|
| `tree` (default) | `cpu` | per-particle Barnes–Hut walk, double precision |
| `group` | `cpu` or `metal` | grouped walk: one walk per group of ≤ `group_size` (32) nearby particles builds an interaction list that every particle in the group evaluates, in single precision |
| `direct` | `cpu` or `metal` | O(N²); double on the CPU, tiled single precision kernel on the GPU |

On an Apple M4 Max with a 200,000 particle galaxy (`galactic-tides-bench`):

| solver | θ | ms per force evaluation | 99th percentile force error |
|--------|---|------------------------:|----------------------------:|
| `tree` on the CPU | 0.6 | 118 | 9.0 × 10⁻³ |
| `group` on Metal | 0.75 | 24 | 5.9 × 10⁻³ |
| `direct` on Metal | – | 122 | 1.8 × 10⁻⁵ |

The grouped walk (Barnes 1990) does 10–30× fewer tree walks and gives every
thread in a GPU SIMD group the same work. Its opening test is the
per-particle one applied to the group's bounding box, so it is never less
accurate; in practice it is about twice as accurate as `tree` at the same θ
(more so in the tail), which is why it can run at θ = 0.75.

The tree and the lists are built on the CPU in double precision; the lists
are evaluated in single precision relative to the root cell, which is ample
for collisionless forces. On the GPU each group runs on one SIMD group: the
lanes fetch list entries in parallel and pass them round with
`simd_shuffle`. Small sibling groups are packed together on the GPU so that
the 32 lanes are mostly busy. Only siblings are merged, so a packed group
stays inside its parent cell. Merging any neighbours in tree order once
produced a group spanning the halo with a list of ~187,000 entries; since
each list is walked serially by one SIMD group, that one group set the
kernel's time. The arrays live in page-aligned host memory that Metal wraps
in place (unified memory), so nothing is copied.

`galactic-tides-bench n=200000` times the solvers and checks them against double
precision direct summation on a sample. For the GPU tree it also shows the
time on the GPU itself, the group fill, the longest list against the mean,
and the GPU's interaction rate.

**Building with Metal.** On macOS, Metal is on by default
(`-DGALACTIC_TIDES_USE_METAL=OFF` to disable). Kernels are compiled at run time
from source embedded in the library, so only the Metal framework is needed.
With full Xcode installed, `cmake --build build --target metal_check`
compiles them offline to surface any syntax errors early.

**How the GPU code is checked.** All of the arithmetic lives in
`include/galactic_tides/kernels/gravity_kernels.h`, written in the common subset
of Metal and C++, so the CPU runs exactly what the GPU runs. On any platform:

* `test_gpu_kernels` checks struct layouts against Metal's, lints everything
  Metal will compile (unsuffixed float literals, `double`, `std::`,
  references), compares the kernels with double precision, and checks the
  interaction lists exactly: every body is covered once per group, no
  accepted cell contains one of the group's own targets, packed groups are
  runs of siblings, and no list dwarfs the mean.
* `test_metal_mock` compiles `src/metal/gravity_kernels.metal` itself as C++
  against a mock `metal_stdlib` and runs it with one CPU thread per GPU
  thread: real barriers and shared threadgroup memory for the brute-force
  kernel, `simd_shuffle` emulated with per-SIMD-group barriers for the tree
  kernel. Results must be bit-identical to the CPU path, and a thread that
  leaves while others still have barriers or shuffles to reach is reported.
  Removing a barrier, shifting an index, reading the wrong lane or letting
  idle threads leave early all make it fail.

On a Mac, `test_metal` then runs the real kernels on the GPU against the CPU.
What only a Mac can check is Metal-specific semantics (address spaces,
attributes, fast-math) and the small Objective-C++ host file.

Limits: 32-bit indices (fine to ~4 × 10⁹ particles), interaction lists held
in memory (a few hundred MB at 10⁶ particles), and macOS may kill a single
very long GPU command, so brute force is for N up to ~10⁵.

## Watching a run in 3D

`viewer/galactic_tides_viewer.html` is a self-contained WebGL page (no build step,
no dependencies) for playing a run back: step or scrub through snapshots,
play at a chosen speed with interpolation between snapshots, orbit, zoom and
pan, jump to face-on or edge-on, and toggle halo, disk and bulge.

```sh
python3 ../scripts/export_viewer.py galaxy/snap_*.hdf5 -o galaxy.gtv
open ../viewer/galactic_tides_viewer.html          # then drop galaxy.gtv on it

# or a single shareable page with the run embedded (opens fine on a phone)
python3 ../scripts/export_viewer.py encounter/snap_*.hdf5 --every 2 --html encounter.html
```

Particles are matched across snapshots by ID. The halo is subsampled to 20%
by default (`--fraction halo=0.5` to change it) and positions are stored as
16 bit integers, so 10k particles over 100 snapshots is about 6 MB.

Keys: space plays and pauses, ← and → step, Home and End jump, F face-on,
E edge-on, R resets the view. Drag rotates, scroll or pinch zooms, shift-drag
or two fingers pan.

## Units

G = 1, disk mass = 1, disk scale length = 1. Scaled to the Milky Way
(h = 3.5 kpc, M_disk = 5.6 × 10¹⁰ M☉): length 3.5 kpc, velocity 262 km/s,
time 13.0 Myr. The disk rotation period at the solar radius (R = 2.43) is
about 16 time units, ~210 Myr.

## What's inside

**Tree** (`octree.h`). Octree with bucket leaves (default 8 particles), built
by recursive counting sort; particles are stored in tree order. Cells carry
monopole and traceless quadrupole moments about their centre of mass
(Hernquist 1987), combined upwards with the parallel axis theorem. The cell
acceptance test is Barnes's offset criterion, accept if d > l/θ + δ with δ
the distance between a cell's centre of mass and its geometric centre, and a
cell containing the target is always opened. Plummer softening. The same
tree answers fixed-radius neighbour queries (`for_each_within`), which is what
SPH and collision detection will need.

Typical accuracy from the test suite (lopsided Hernquist sphere plus
satellites): median relative force error ~4 × 10⁻⁴ at θ = 0.5 and
~3 × 10⁻³ at θ = 1.0 with quadrupoles.

**Integrator** (`integrator.h`). Kick-drift-kick leapfrog with a global
timestep. Forces come from a `ForceModel`; `TreeGravity` is the only one so
far. The run log reports energy drift, virial ratio, momentum, L_z and
interactions per particle.

**Initial conditions** (`hernquist_ics.h`). Hernquist (1993), "N-body
realizations of compound galaxies", ApJS 86, 389:

| component | density | velocities |
|-----------|---------|------------|
| disk | Σ ∝ exp(−R/h), sech²(z/z₀) | σ_z² = πGΣz₀; σ_R² ∝ exp(−R/h) normalised to Toomre Q at R = 2.43h; σ_φ from the epicyclic approximation; mean rotation from asymmetric drift |
| bulge | Hernquist (1990), ρ ∝ a / r(r+a)³ | isotropic Gaussian, spherical Jeans σ, truncated at 0.95 v_esc |
| halo | ρ ∝ exp(−r²/r_c²)/(r²+γ²) | as for the bulge |

Default masses and scales are Hernquist's Milky-Way-like model (M_halo = 5.8,
M_bulge = 1/3, γ = 1, r_c = 10, a = 0.2, z₀ = 0.2, Q = 1.5; worth checking
against the paper if you want an exact reproduction). Every force used in the
moment equations — the spheroids' radial force, the disk's rotation curve and
epicyclic frequency — is measured with the softened tree on the realised
particles and angle-averaged, so the dispersions are consistent with the
forces the run will apply.

The scheme is approximate (Maxwellians are not equilibrium distribution
functions; Kazantzidis, Magorrian & Moore 2004). Measured with the tree, the
default recipe leaves the halo about 8% cold, because truncating the Gaussian
at the escape speed discards its tail, and leaves the disk vertically cold,
because the isothermal sheet ignores the halo and bulge's vertical pull. Two
optional refinements, off by default:

* `preserve_dispersion = true` widens the Gaussian so the truncated
  distribution keeps the Jeans second moment;
* `disk_sigma_z = jeans` solves the vertical Jeans equation in the full
  potential.

With both on, each component starts within a few percent of virial
equilibrium. Running the two variants side by side is a nice way to watch IC
relaxation (`plot_snapshot.py profiles`).

**Snapshots** (`snapshot_io.h`). GADGET-layout HDF5: `Header` attributes and
`PartTypeN` groups with `Coordinates`, `Velocities`, `Masses`,
`ParticleIDs`, `Potential`. Types follow GADGET: 0 gas (reserved),
1 halo, 2 disk, 3 bulge. pynbody and yt read these, and ICs in the same layout
from other generators (DICE, MakeNewDisk, AGAMA after conversion) can be run
directly. Each snapshot records the parameters that produced it as string
attributes on `/Parameters`.

## Layout

```
include/galactic_tides/   vec3, particles, params, octree, gravity, integrator,
                     snapshot_io, hernquist_ics, group_lists, gravity_backends,
                     metal_backend, kernels/gravity_kernels.h (CPU + Metal)
src/                 implementations; src/metal: kernels and Metal host code
apps/                galactic-tides-ics, galactic-tides-run, galactic-tides-bench
tests/               self-contained tests (ctest)
scripts/             plot_snapshot.py (image / profiles / frames),
                     export_viewer.py (pack snapshots for the viewer)
viewer/              galactic_tides_viewer.html (WebGL playback)
examples/            parameter files
```

## Room to grow

* **SPH.** Gas is `ParticleType::Gas`; neighbour finding via
  `Octree::for_each_within`; an `SphForces` model alongside `TreeGravity`
  (TREESPH, Hernquist & Katz 1989, is the natural lineage). A gas disk in
  the ICs following Springel, Di Matteo & Hernquist (2005).
* **Collisions.** Ball queries on the tree give candidate pairs; sticky
  particle cloud collisions are a classic cheap ISM model, and merging
  particles into sinks (`ParticleType::Sink`) is another.
* **Timesteps.** Individual (block) timesteps would help the centre a lot.
* **Softening.** Spline softening (Monaghan & Lattanzio) in place of Plummer.
* **Better ICs.** Distribution-function based models (Kuijken & Dubinski 1995;
  AGAMA) for comparison with the moment-based ones.

## References

* Barnes, J. E. 1990, J. Comput. Phys., 87, 161
* Barnes, J. & Hut, P. 1986, Nature, 324, 446
* Barnes, J. E. & Hernquist, L. 1992, ARA&A, 30, 705
* Hernquist, L. 1987, ApJS, 64, 715
* Hernquist, L. 1990, ApJ, 356, 359
* Hernquist, L. 1993, ApJS, 86, 389
* Hernquist, L. & Katz, N. 1989, ApJS, 70, 419
* Kazantzidis, S., Magorrian, J. & Moore, B. 2004, ApJ, 601, 37
* Kuijken, K. & Dubinski, J. 1995, MNRAS, 277, 1341
* Springel, V., Di Matteo, T. & Hernquist, L. 2005, MNRAS, 361, 776
* Toomre, A. & Toomre, J. 1972, ApJ, 178, 623
