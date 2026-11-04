#!/usr/bin/env python3
"""Quick-look plots of galactic-tides snapshots (GADGET-style HDF5).

  plot_snapshot.py image snap_0000.hdf5 [snap_0010.hdf5 ...] [-o out.png]
      Face-on and edge-on stellar (disk + bulge) surface density, one row
      per snapshot. Use --halo to include the dark matter, --extent to zoom.

  plot_snapshot.py profiles snap_0000.hdf5 snap_0030.hdf5 [-o out.png]
      Disk radial profiles for each snapshot: surface density, mean rotation,
      velocity dispersions and thickness. Good for watching the initial
      conditions relax (or not).

  plot_snapshot.py frames output/snap_*.hdf5 --outdir frames
      One image per snapshot, for stitching into a movie, e.g.
      ffmpeg -framerate 12 -i frames/frame_%04d.png -pix_fmt yuv420p movie.mp4

Needs numpy, h5py and matplotlib.
"""

import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

TYPES = {"gas": 0, "halo": 1, "disk": 2, "bulge": 3, "stars": 4, "sink": 5}


def load(path, kinds):
    """Positions, velocities and masses of the requested particle kinds."""
    pos, vel, mass = [], [], []
    with h5py.File(path, "r") as f:
        time = float(f["Header"].attrs.get("Time", 0.0))
        for k in kinds:
            g = f"PartType{TYPES[k]}"
            if g not in f:
                continue
            pos.append(f[g]["Coordinates"][:])
            vel.append(f[g]["Velocities"][:])
            mass.append(f[g]["Masses"][:])
    if not pos:
        return time, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    return time, np.concatenate(pos), np.concatenate(vel), np.concatenate(mass)


def centre(pos, mass, vel=None, iterations=5, radius=5.0):
    """Shrinking-sphere centre, so that offset or merging systems stay framed."""
    c = np.average(pos, axis=0, weights=mass)
    r = radius * 4
    for _ in range(iterations):
        sel = np.linalg.norm(pos - c, axis=1) < r
        if sel.sum() < 50:
            break
        c = np.average(pos[sel], axis=0, weights=mass[sel])
        r = max(radius, r * 0.7)
    v = None
    if vel is not None:
        sel = np.linalg.norm(pos - c, axis=1) < radius
        v = np.average(vel[sel], axis=0, weights=mass[sel]) if sel.any() else vel.mean(0)
    return c, v


def blur(img, sigma):
    """Separable Gaussian blur (sigma in pixels), numpy only."""
    if sigma <= 0:
        return img
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    img = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, img)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, img)


def surface_density(ax, x, y, m, extent, bins, smooth, vmax, dynamic_range):
    h, xe, ye = np.histogram2d(
        x, y, bins=bins, range=[[-extent, extent], [-extent, extent]], weights=m
    )
    h = blur(h / ((xe[1] - xe[0]) * (ye[1] - ye[0])), smooth)
    if vmax is None:
        vmax = np.percentile(h[h > 0], 99.8) if (h > 0).any() else 1.0
    h = np.where(h > vmax / dynamic_range, h, np.nan)
    im = ax.imshow(
        h.T,
        origin="lower",
        extent=[-extent, extent, -extent, extent],
        norm=LogNorm(vmin=vmax / dynamic_range, vmax=vmax),
        cmap="magma",
        interpolation="bilinear",
    )
    ax.set_facecolor("black")
    return im, vmax


def cmd_image(args, paths=None, out=None):
    paths = paths or args.snapshots
    kinds = ["disk", "bulge"] + (["halo"] if args.halo else [])
    fig, axes = plt.subplots(len(paths), 2, figsize=(10, 4.6 * len(paths)), squeeze=False,
                             layout="constrained")
    for row, path in zip(axes, paths):
        t, pos, _, m = load(path, kinds)
        if args.recentre:
            c, _ = centre(pos, m)
            pos = pos - c
        ext = args.extent
        _, vmax = surface_density(row[0], pos[:, 0], pos[:, 1], m, ext, args.bins, args.smooth,
                                  args.vmax, args.range)
        im, _ = surface_density(row[1], pos[:, 0], pos[:, 2], m, ext, args.bins, args.smooth,
                                vmax, args.range)
        row[0].set_title(f"{os.path.basename(path)}  t = {t:.2f}   face-on (x-y)")
        row[1].set_title("edge-on (x-z)")
        for ax in row:
            ax.set_xlabel("x")
        row[0].set_ylabel("y")
        row[1].set_ylabel("z")
        fig.colorbar(im, ax=row, shrink=0.8, label="surface density")
    fig.savefig(out or args.output, dpi=args.dpi)
    plt.close(fig)


def cmd_frames(args):
    os.makedirs(args.outdir, exist_ok=True)
    for i, path in enumerate(args.snapshots):
        out = os.path.join(args.outdir, f"frame_{i:04d}.png")
        cmd_image(args, [path], out)
        print(out)


def disk_profiles(path, rmax, nbins):
    t, pos, vel, m = load(path, ["disk"])
    _, pall, _, mall = load(path, ["disk", "bulge", "halo"])
    c, _ = centre(pall, mall)
    pos = pos - c
    sel = np.linalg.norm(pos, axis=1) < 3 * rmax
    vel = vel - np.average(vel[sel], axis=0, weights=m[sel])
    # Align z with the disk's angular momentum, in case it has been tilted.
    L = np.sum(m[:, None] * np.cross(pos, vel), axis=0)
    ez = L / np.linalg.norm(L)
    ex = np.cross([0, 1, 0], ez) if abs(ez[1]) < 0.9 else np.cross([1, 0, 0], ez)
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    x, y, z = pos @ ex, pos @ ey, pos @ ez
    vx, vy, vz = vel @ ex, vel @ ey, vel @ ez
    R = np.hypot(x, y)
    vR = (x * vx + y * vy) / np.maximum(R, 1e-12)
    vphi = (x * vy - y * vx) / np.maximum(R, 1e-12)
    edges = np.linspace(0, rmax, nbins + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(R, edges) - 1
    out = {k: np.full(nbins, np.nan) for k in ("sigma", "vphi", "sR", "sphi", "sz", "zrms")}
    for i in range(nbins):
        s = idx == i
        if s.sum() < 10:
            continue
        out["sigma"][i] = m[s].sum() / (np.pi * (edges[i + 1] ** 2 - edges[i] ** 2))
        out["vphi"][i] = vphi[s].mean()
        out["sR"][i] = vR[s].std()
        out["sphi"][i] = vphi[s].std()
        out["sz"][i] = vz[s].std()
        out["zrms"][i] = np.sqrt(np.mean(z[s] ** 2))
    return t, mid, out


def cmd_profiles(args):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    panels = [
        ("sigma", "surface density", True),
        ("vphi", "mean v_phi", False),
        ("zrms", "rms z", False),
        ("sR", "sigma_R", False),
        ("sphi", "sigma_phi", False),
        ("sz", "sigma_z", False),
    ]
    for path in args.snapshots:
        t, R, prof = disk_profiles(path, args.rmax, args.nbins)
        for ax, (key, label, logy) in zip(axes.flat, panels):
            ax.plot(R, prof[key], label=f"t = {t:.1f}")
            ax.set_xlabel("R")
            ax.set_title(label)
            if logy:
                ax.set_yscale("log")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("snapshots", nargs="+")
    common.add_argument("--dpi", type=int, default=110)
    img = argparse.ArgumentParser(add_help=False)
    img.add_argument("--extent", type=float, default=10.0, help="half width of the view")
    img.add_argument("--bins", type=int, default=256)
    img.add_argument("--smooth", type=float, default=1.5, help="Gaussian blur, pixels")
    img.add_argument("--vmax", type=float, default=None,
                     help="top of the colour scale (default: from the face-on image); "
                          "fix it when making movie frames")
    img.add_argument("--range", type=float, default=1e4, help="colour scale dynamic range")
    img.add_argument("--halo", action="store_true", help="include halo particles")
    img.add_argument("--no-recentre", dest="recentre", action="store_false")

    s = sub.add_parser("image", parents=[common, img])
    s.add_argument("-o", "--output", default="snapshot.png")
    s = sub.add_parser("frames", parents=[common, img])
    s.add_argument("--outdir", default="frames")
    s = sub.add_parser("profiles", parents=[common])
    s.add_argument("-o", "--output", default="profiles.png")
    s.add_argument("--rmax", type=float, default=8.0)
    s.add_argument("--nbins", type=int, default=32)

    args = p.parse_args()
    {"image": cmd_image, "frames": cmd_frames, "profiles": cmd_profiles}[args.cmd](args)


if __name__ == "__main__":
    main()
