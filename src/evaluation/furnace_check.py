"""
Checks a rendered white-furnace EXR (furnace_alum.rib / furnace_chrome.rib)
against the no-renderer directional-albedo curve from compute_albedo.py.

Under a uniform white dome of radiance 1.0, the orthographic camera sees,
at the point on the sphere whose view angle from the normal is theta_o,
exactly rho(theta_o) -- the same quantity compute_albedo.py computes
directly from the network weights with no renderer involved. If the
rendered pixel values (averaged in radial bins) match that curve, the
plugin's actual sampler + MIS path is reproducing the un-rendered
reference -- the thing the raw-weights albedo check (A1 part 1) cannot
test on its own.

Geometry assumed (matches furnace_alum.rib / furnace_chrome.rib defaults):
  orthographic camera, ScreenWindow [-screen_half, screen_half]^2, square
  image of `res` x `res` pixels, sphere of `sphere_radius` at the origin.
  Pass --screen-half / --res / --sphere-radius if you changed the RIB.

Usage:
    python furnace_check.py --exr furnace_alum.exr --albedo-csv albedo_alum.csv \
        --out furnace_alum_compare.csv

    python furnace_check.py --self-test     # no files needed

Requires: numpy, opencv-python (same env as your other EXR scripts).
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse
import csv
import sys
import numpy as np
import cv2

# Defaults must match furnace_alum.rib / furnace_chrome.rib.
RES = 512
SCREEN_HALF = 1.2
SPHERE_RADIUS = 1.0
N_BINS = 8


def load_exr_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img[..., :3][..., ::-1].astype(np.float64)   # BGR -> RGB

    # cv2 failed outright -- on at least one RenderMan install in this
    # project, `Display "x.exr" "file" "rgba"` actually writes an
    # LZW-compressed float TIFF with a ".exr" filename (confirmed via the
    # file's own magic bytes: "II*\x00", not OpenEXR's). OpenCV's bundled
    # libtiff can't decode that LZW stream even with
    # OPENCV_IO_ENABLE_OPENEXR=1 (that flag only affects real EXR
    # decoding). Fall back to tifffile+imagecodecs, which can.
    print(f"NOTE: cv2 could not decode {path} as OpenEXR; it may actually be "
          f"a TIFF with a '.exr' name (check with "
          f"`open(path,'rb').read(8)` -- TIFF starts with b'II*\\x00' or "
          f"b'MM\\x00*'). Falling back to tifffile.", file=sys.stderr)
    try:
        import tifffile
    except ImportError:
        raise SystemExit(
            f"could not load {path}: cv2 failed and tifffile is not "
            f"installed for the TIFF-as-.exr fallback "
            f"(pip install tifffile imagecodecs)")
    img = tifffile.imread(path)
    if img is None:
        raise SystemExit(f"could not load {path} with either cv2 or tifffile")
    # tifffile preserves the TIFF's own sample order; PRMan's "rgba"
    # display writes true R,G,B,(A) order (unlike cv2's BGR convention
    # above), so no channel reversal here.
    return img[..., :3].astype(np.float64)


def pixel_theta_o(res, screen_half, sphere_radius):
    """Per-pixel theta_o (radians) and a boolean mask of pixels inside the
    sphere's silhouette, from the orthographic camera geometry alone (no
    dependence on rendered pixel values)."""
    px_per_unit = res / (2.0 * screen_half)
    idx = np.arange(res)
    # column -> x (world), row -> y (world); row 0 is image top = +y,
    # matching the convention used in render_gt_comparison.py.
    x = (idx - (res - 1) / 2.0) / px_per_unit
    y = -(idx - (res - 1) / 2.0) / px_per_unit
    X, Y = np.meshgrid(x, y)
    r_world = np.sqrt(X**2 + Y**2) / sphere_radius
    inside = r_world <= 1.0
    theta_o = np.full_like(r_world, np.nan)
    theta_o[inside] = np.arcsin(np.clip(r_world[inside], 0.0, 1.0))
    return theta_o, inside


def radial_bin(img_rgb, theta_o, inside, n_bins):
    """Mean RGB per radial (theta_o) bin, over sphere pixels only."""
    edges = np.linspace(0.0, np.pi / 2, n_bins + 1)
    centers_deg = np.rad2deg((edges[:-1] + edges[1:]) / 2.0)
    means = np.zeros((n_bins, 3))
    counts = np.zeros(n_bins, dtype=int)
    th = theta_o[inside]
    px = img_rgb[inside]
    bin_idx = np.clip(np.digitize(th, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        sel = bin_idx == b
        counts[b] = sel.sum()
        if sel.any():
            means[b] = px[sel].mean(axis=0)
    return centers_deg, means, counts


def load_rho_csv(path, source="NBRDF"):
    """Read compute_albedo.py's --csv output; return angle->rgb for one source."""
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["source"] == source:
                rows.append((float(row["theta_o_deg"]),
                             float(row["rhoR"]), float(row["rhoG"]), float(row["rhoB"])))
    if not rows:
        raise SystemExit(f"no rows with source='{source}' in {path}")
    rows.sort()
    angles = np.array([r[0] for r in rows])
    rgb = np.array([r[1:] for r in rows])
    return angles, rgb


def interp_rho(angles, rgb, query_deg):
    """Per-channel linear interpolation of the (angle -> rgb) reference curve."""
    out = np.zeros((len(query_deg), 3))
    for c in range(3):
        out[:, c] = np.interp(query_deg, angles, rgb[:, c])
    return out


def compare(exr_path, albedo_csv, res, screen_half, sphere_radius, n_bins,
            source="NBRDF", out_csv=None, background_check=True):
    img = load_exr_rgb(exr_path)
    if img.shape[0] != res or img.shape[1] != res:
        print(f"WARNING: EXR is {img.shape[1]}x{img.shape[0]}, expected {res}x{res} "
              f"-- geometry mapping below will be wrong unless you pass matching --res.")

    theta_o, inside = pixel_theta_o(res, screen_half, sphere_radius)

    if background_check:
        bg = img[~inside]
        if bg.size:
            bg_mean = bg.mean(axis=0)
            print(f"background mean (expect ~1.0, dome directly visible): "
                  f"R={bg_mean[0]:.4f} G={bg_mean[1]:.4f} B={bg_mean[2]:.4f}")
            if bg_mean.max() < 0.5:
                print("  WARNING: background is dim, not ~1.0 -- dome light may not be "
                      "reaching the camera either; check the render before trusting the "
                      "sphere comparison below.")

    centers_deg, rendered_rgb, counts = radial_bin(img, theta_o, inside, n_bins)
    ref_angles, ref_rgb = load_rho_csv(albedo_csv, source=source)
    ref_at_centers = interp_rho(ref_angles, ref_rgb, centers_deg)

    diff = rendered_rgb - ref_at_centers
    rmse_per_channel = np.sqrt(np.mean(diff**2, axis=0))
    max_abs = np.max(np.abs(diff))

    print(f"\n{os.path.basename(exr_path)} vs {source} rho-curve in {albedo_csv}")
    print(f"{'theta_o':>8} | {'px R':>7} {'px G':>7} {'px B':>7} | "
          f"{'ref R':>7} {'ref G':>7} {'ref B':>7} | {'npix':>6}")
    rows_out = []
    for i, a in enumerate(centers_deg):
        print(f"{a:8.1f} | {rendered_rgb[i,0]:7.4f} {rendered_rgb[i,1]:7.4f} {rendered_rgb[i,2]:7.4f} | "
              f"{ref_at_centers[i,0]:7.4f} {ref_at_centers[i,1]:7.4f} {ref_at_centers[i,2]:7.4f} | "
              f"{counts[i]:6d}")
        rows_out.append([a, *rendered_rgb[i], *ref_at_centers[i], counts[i]])

    print(f"\nRMSE per channel: R={rmse_per_channel[0]:.5f} G={rmse_per_channel[1]:.5f} "
          f"B={rmse_per_channel[2]:.5f}   max|diff|={max_abs:.5f}")
    # Loose default bar: a few percent of a typical rho magnitude (~0.1-0.25
    # per the alum-bronze/chrome numbers already measured). Tighten once you
    # have a feel for sampling noise at your chosen sample count.
    ok = max_abs < 0.02
    print(f"  {'PASS' if ok else 'CHECK'}: max|diff| {'<' if ok else '>='} 0.02 "
          f"({'renderer matches the un-rendered reference' if ok else 'investigate -- see protocol A2/A3 next'})")

    if out_csv:
        with open(out_csv, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["theta_o_deg", "renderR", "renderG", "renderB",
                         "refR", "refG", "refB", "npix"])
            wr.writerows(rows_out)
        print(f"Wrote {out_csv}")

    return centers_deg, rendered_rgb, ref_at_centers


def self_test():
    """Synthesize a furnace EXR for a KNOWN analytic rho(theta) curve, write
    a matching albedo CSV, run compare(), and check recovered values track
    the known curve closely."""
    import tempfile

    def true_rho(theta_rad):
        # Smooth, bounded, nothing special -- just something to recover.
        t = np.asarray(theta_rad)
        base = 0.08 + 0.12 * np.sin(t) ** 2
        return np.stack([base, 0.8 * base, 0.6 * base], axis=-1)

    res, screen_half, sphere_r = 256, 1.2, 1.0
    theta_o, inside = pixel_theta_o(res, screen_half, sphere_r)
    img = np.ones((res, res, 3), np.float32)        # background = 1.0 (dome)
    img[inside] = true_rho(theta_o[inside]).astype(np.float32)
    rng = np.random.default_rng(0)
    img[inside] += rng.normal(0, 0.002, img[inside].shape)  # tiny sampling noise

    d = tempfile.mkdtemp()
    exr_path = os.path.join(d, "furnace_self_test.exr")
    try:
        ok = cv2.imwrite(exr_path, img[..., ::-1])
    except cv2.error:
        ok = False
    if not ok:
        # This OpenCV build has no OpenEXR writer compiled in (the
        # OPENCV_IO_ENABLE_OPENEXR env var only gates *decoding*, not
        # writing). Fall back to a float TIFF saved with a ".exr" name --
        # load_exr_rgb() above already knows how to read exactly that,
        # because some real RenderMan installs write files this way too.
        import tifffile
        tifffile.imwrite(exr_path, img.astype(np.float32))

    csv_path = os.path.join(d, "albedo_self_test.csv")
    angles = np.array([0, 15, 30, 45, 60, 70, 80, 85], float)
    with open(csv_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["source", "theta_o_deg", "rhoR", "rhoG", "rhoB"])
        for a in angles:
            r = true_rho(np.deg2rad(a))
            wr.writerow(["NBRDF", a, f"{r[0]:.6f}", f"{r[1]:.6f}", f"{r[2]:.6f}"])

    centers_deg, rendered_rgb, ref_rgb = compare(
        exr_path, csv_path, res, screen_half, sphere_r, N_BINS,
        source="NBRDF", out_csv=None, background_check=True)

    err = np.max(np.abs(rendered_rgb - ref_rgb))
    status = "OK" if err < 0.01 else "FAIL"
    print(f"\nself-test max|diff|={err:.5f} -> {status} (expect < 0.01)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exr", help="rendered furnace EXR (e.g. furnace_alum.exr)")
    ap.add_argument("--albedo-csv", help="CSV from compute_albedo.py's --csv output")
    ap.add_argument("--source", default="NBRDF", choices=["NBRDF", "GT"],
                    help="which curve in the CSV to compare against (default NBRDF: "
                         "tests the render against the network's own un-rendered "
                         "albedo -- the correct Track A1 comparison)")
    ap.add_argument("--res", type=int, default=RES)
    ap.add_argument("--screen-half", type=float, default=SCREEN_HALF)
    ap.add_argument("--sphere-radius", type=float, default=SPHERE_RADIUS)
    ap.add_argument("--bins", type=int, default=N_BINS)
    ap.add_argument("--out", help="optional CSV of the per-bin comparison")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.exr or not args.albedo_csv:
        ap.error("need --exr and --albedo-csv (or --self-test)")

    compare(args.exr, args.albedo_csv, args.res, args.screen_half, args.sphere_radius,
            args.bins, source=args.source, out_csv=args.out)


if __name__ == "__main__":
    main()
