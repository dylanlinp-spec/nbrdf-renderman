"""
Track B (performance-evaluation-protocol.md): sampler-efficiency, the
metric furnace_check.py cannot show.

furnace_check.py bins thousands of pixels together and reports the BIN
MEAN vs. the analytic albedo curve. At any spp, that mean is dominated
by the systematic bias already documented for Track A (the still-open
energy-conservation mismatch, ~0.1-0.3 magnitude) -- which swamps the
comparatively tiny noise difference between sampling techniques. That's
why mixture/cosine-only/GGX-only came back bit-identical at both 512
and 16 spp: changing spp-per-pixel doesn't matter when each bin already
averages over tens of thousands of samples spatially.

What actually separates samplers is per-pixel NOISE (variance), not the
bin mean. This script reports, per angular bin, the standard deviation
of individual pixel values (not their mean) -- i.e. how much pixels at
the same theta_o disagree with EACH OTHER. A noisier sampler (GGX-only,
expected worse per Sec 3.2) should show a visibly higher std-dev than
the mixture or cosine-only at the same low spp. This is the Track B
number: it isolates sampler efficiency from the Track A bias instead of
being swallowed by it.

Usage (same three renders you already have):
    python track_b_noise.py --exr b16_furnace_alum_mixture.exr --label mixture
    python track_b_noise.py --exr b16_furnace_alum_cosine.exr  --label cosine-only
    python track_b_noise.py --exr b16_furnace_alum_ggx.exr     --label GGX-only
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse
import sys
import numpy as np
import cv2


def load_exr_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img[..., :3][..., ::-1].astype(np.float64)
    print(f"NOTE: cv2 could not decode {path}; falling back to tifffile.", file=sys.stderr)
    try:
        import tifffile
    except ImportError:
        raise SystemExit(f"could not load {path}: install tifffile+imagecodecs")
    img = tifffile.imread(path)
    if img is None:
        raise SystemExit(f"could not load {path} with either cv2 or tifffile")
    return img[..., :3].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exr", required=True)
    ap.add_argument("--label", required=True, help="e.g. mixture / cosine-only / GGX-only")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--screen-half", type=float, default=1.2)
    ap.add_argument("--sphere-radius", type=float, default=1.0)
    ap.add_argument("--bins", type=int, default=8)
    args = ap.parse_args()

    img = load_exr_rgb(args.exr)
    if img.shape[:2] != (args.res, args.res):
        raise SystemExit(f"shape {img.shape[:2]} != expected ({args.res},{args.res}) -- "
                          f"pass --res to match")

    # Same orthographic-sphere pixel->theta_o geometry as furnace_check.py.
    coords = np.linspace(-args.screen_half, args.screen_half, args.res)
    xx, yy = np.meshgrid(coords, coords)
    yy = yy[::-1]  # image row 0 is top = +y
    r2 = xx**2 + yy**2
    on_sphere = r2 <= args.sphere_radius**2
    zz = np.sqrt(np.clip(args.sphere_radius**2 - r2, 0, None))
    theta_o = np.degrees(np.arccos(np.clip(zz / args.sphere_radius, -1, 1)))

    edges = np.linspace(0, 90, args.bins + 1)
    print(f"\n{os.path.basename(args.exr)}  [{args.label}]  "
          f"per-bin pixel-to-pixel noise (std-dev), NOT compared to reference")
    print(f"{'theta_o':>8} | {'std R':>8} {'std G':>8} {'std B':>8} | {'mean R':>8} {'npix':>7}")
    for i in range(args.bins):
        lo, hi = edges[i], edges[i+1]
        mask = on_sphere & (theta_o >= lo) & (theta_o < hi)
        n = int(mask.sum())
        if n < 2:
            continue
        pix = img[mask]
        std = pix.std(axis=0)
        mean = pix.mean(axis=0)
        mid = 0.5 * (lo + hi)
        print(f"{mid:8.1f} | {std[0]:8.4f} {std[1]:8.4f} {std[2]:8.4f} | "
              f"{mean[0]:8.4f} {n:7d}")

    overall_std = img[on_sphere].std(axis=0)
    print(f"\nOverall sphere pixel std-dev (noise proxy): "
          f"R={overall_std[0]:.5f} G={overall_std[1]:.5f} B={overall_std[2]:.5f}")
    print("Compare this line across mixture/cosine-only/GGX-only at the SAME spp -- "
          "the technique with the lowest std-dev here is the more sample-efficient one. "
          "This is the Track B number (independent of the Track A bias, which affects "
          "all three equally and cancels out of a std-dev).")


if __name__ == "__main__":
    main()
