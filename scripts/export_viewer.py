#!/usr/bin/env python3
"""Pack a run's snapshots for the 3D viewer (viewer/galactic_tides_viewer.html).

  export_viewer.py output/snap_*.hdf5 -o run.gtv
      Writes a compact binary file; open the viewer page in a browser and
      drop the file on it (or use its "Open run" button).

  export_viewer.py output/snap_*.hdf5 --html run.html
      Writes a single self-contained HTML page with the run embedded: handy
      for sharing or for opening on a phone.

Particles are matched across snapshots by ParticleIDs, so every frame has the
same particles in the same order and the viewer can interpolate between
frames. To keep files small the halo is subsampled by default
(--fraction halo=0.2); positions are stored as 16 bit integers scaled to the
run's extent (particles flung beyond it are hidden rather than clamped).

File format (little endian): b"GTV1", uint32 header length, UTF-8 JSON header
padded with spaces to a multiple of 4 bytes, then int16[n_frames][n_particles][3].
The value -32768 in x marks a particle that is absent or out of range.
"""

import argparse
import base64
import json
import os
import struct
import sys

import h5py
import numpy as np

TYPES = {0: "gas", 1: "halo", 2: "disk", 3: "bulge", 4: "stars", 5: "sink"}
HIDDEN = -32768
MAX_Q = 32767


def read(path):
    """time, {type: (ids, positions)} with each type sorted by ID."""
    out = {}
    with h5py.File(path, "r") as f:
        time = float(f["Header"].attrs.get("Time", 0.0))
        for t, name in TYPES.items():
            g = f.get(f"PartType{t}")
            if g is None:
                continue
            pos = g["Coordinates"][:].astype(np.float64)
            ids = g["ParticleIDs"][:] if "ParticleIDs" in g else np.arange(len(pos))
            order = np.argsort(ids, kind="stable")
            out[name] = (ids[order], pos[order])
    return time, out


def parse_fractions(items):
    frac = {"halo": 0.2}
    for item in items or []:
        name, _, value = item.partition("=")
        if name not in TYPES.values() or not value:
            sys.exit(f"--fraction expects TYPE=FRACTION with TYPE one of {list(TYPES.values())}")
        frac[name] = float(value)
    return frac


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("snapshots", nargs="+")
    p.add_argument("-o", "--output", help="binary .gtv file")
    p.add_argument("--html", help="self-contained HTML page with the run embedded")
    p.add_argument("--viewer", default=os.path.join(os.path.dirname(__file__), "..", "viewer",
                                                    "galactic_tides_viewer.html"),
                   help="viewer template for --html")
    p.add_argument("--fraction", action="append", metavar="TYPE=F",
                   help="keep this fraction of a particle type (default halo=0.2, others 1)")
    p.add_argument("--every", type=int, default=1, help="use every Nth snapshot")
    p.add_argument("--extent", type=float, default=None,
                   help="half width of the stored volume (default: from the data)")
    p.add_argument("--title", default=None)
    p.add_argument("--time-unit-myr", type=float, default=13.05,
                   help="Myr per time unit for the readout (Milky Way scaling: 13.05)")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if not args.output and not args.html:
        p.error("give -o/--output and/or --html")

    paths = sorted(args.snapshots)[:: args.every]
    frac = parse_fractions(args.fraction)
    rng = np.random.default_rng(args.seed)

    # Choose particles from the first snapshot.
    t0, first = read(paths[0])
    groups, chosen = [], {}
    offset = 0
    for name in ("halo", "disk", "bulge", "gas", "stars", "sink"):
        if name not in first:
            continue
        ids = first[name][0]
        f = frac.get(name, 1.0)
        keep = ids if f >= 1 else np.sort(rng.choice(ids, max(1, int(round(f * len(ids)))), replace=False))
        if len(keep) == 0:
            continue
        chosen[name] = keep
        groups.append({"name": name, "offset": offset, "count": int(len(keep)),
                       "represents": int(len(ids))})
        offset += len(keep)
    n = offset

    frames, times = [], []
    for path in paths:
        t, snap = read(path)
        xyz = np.full((n, 3), np.nan)
        for g in groups:
            if g["name"] not in snap:
                continue
            ids, pos = snap[g["name"]]
            want = chosen[g["name"]]
            j = np.searchsorted(ids, want)
            j = np.clip(j, 0, max(len(ids) - 1, 0))
            ok = (len(ids) > 0) & (ids[j] == want)
            block = np.full((len(want), 3), np.nan)
            block[ok] = pos[j[ok]]
            xyz[g["offset"]:g["offset"] + g["count"]] = block
        frames.append(xyz)
        times.append(t)
        print(f"{path}: t = {t:g}", file=sys.stderr)

    stack = np.stack(frames)
    finite = np.isfinite(stack).all(axis=2)
    if args.extent:
        extent = args.extent
    else:
        r = np.abs(stack[finite]).max(axis=1)
        extent = float(np.percentile(r, 99.9)) * 1.05 if r.size else 1.0
    q = np.round(stack / extent * MAX_Q)
    bad = ~finite | (np.abs(q) > MAX_Q - 1).any(axis=2)
    # -32767 and -32768 both normalise to -1.0 in WebGL, so keep real values
    # at or above -32766 and reserve the bottom for the hidden marker.
    q = np.nan_to_num(q, nan=0.0).clip(-MAX_Q + 1, MAX_Q).astype("<i2")
    q[bad] = (HIDDEN, 0, 0)

    header = {
        "format": "galactic-tides-viewer",
        "version": 1,
        "title": args.title or os.path.basename(os.path.dirname(os.path.abspath(paths[0]))),
        "n_frames": len(paths),
        "n_particles": n,
        "extent": extent,
        "times": times,
        "time_unit_myr": args.time_unit_myr,
        "groups": groups,
    }
    hjson = json.dumps(header, separators=(",", ":")).encode()
    hjson += b" " * (-len(hjson) % 4)
    blob = b"GTV1" + struct.pack("<I", len(hjson)) + hjson + q.tobytes()
    print(f"{len(paths)} frames, {n} particles, extent {extent:.3g}, "
          f"{bad.sum()} hidden particle-frames, {len(blob) / 1e6:.1f} MB", file=sys.stderr)

    if args.output:
        with open(args.output, "wb") as f:
            f.write(blob)
        print(f"wrote {args.output}", file=sys.stderr)
    if args.html:
        with open(args.viewer, encoding="utf-8") as f:
            page = f.read()
        marker = '<script id="gtv-embedded" type="application/octet-stream"></script>'
        if marker not in page:
            sys.exit(f"{args.viewer} has no embedding placeholder")
        data = base64.b64encode(blob).decode("ascii")
        page = page.replace(marker, marker.replace("></script>", f">{data}</script>"))
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(page)
        print(f"wrote {args.html}", file=sys.stderr)


if __name__ == "__main__":
    main()
