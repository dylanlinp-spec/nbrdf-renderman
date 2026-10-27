"""
Track A2 (performance-evaluation-protocol.md): diffs a real RenderMan
render (a2_distant_<material>.rib) against the analytic, zero-Monte-Carlo
reference dumped by render_gt_comparison.py (<material>_nbrdf_linear.npy
+ <material>_mask.npy), pixel-for-pixel, in LINEAR space (no exposure/
gamma -- that PNG is for viewing only, never diff against it).

Usage:
    python a2_check.py --exr a2_distant_alum.exr --label alum-bronze
    python a2_check.py --exr a2_distant_alum.exr --label alum-bronze --out a2_alum_compare.csv

Requires the same loader as furnace_check.py (cv2, falling back to
tifffile+imagecodecs for this RenderMan install's TIFF-as-.exr output).
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
        return img[..., :3][..., ::-1].astype(np.float64)   # BGR -> RGB
    print(f"NOTE: cv2 could not decode {path}; falling back to tifffile "
          f"(this RenderMan install writes TIFF under a '.exr' name -- "
          f"see furnace_check.py's loader for the full explanation).",
          file=sys.stderr)
    try:
        import tifffile
    except ImportError:
        raise SystemExit(f"could not load {path}: install tifffile+imagecodecs")
    img = tifffile.imread(path)
    if img is None:
        raise SystemExit(f"could not load {path} with either cv2 or tifffile")
    return img[..., :3].astype(np.float64)   # tifffile preserves true RGB(A) order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exr", required=True, help="rendered a2_distant_<material>.exr")
    ap.add_argument("--label", required=True,
                     help="material label matching render_gt_comparison.py's dump "
                          "(e.g. 'alum-bronze' -> reads alum-bronze_nbrdf_linear.npy "
                          "+ alum-bronze_mask.npy)")
    ap.add_argument("--out", help="optional CSV of per-pixel diffs on lit pixels")
    args = ap.parse_args()

    ref = np.load(f"{args.label}_nbrdf_linear.npy")     # (RES,RES,3) linear, black bg
    mask = np.load(f"{args.label}_mask.npy")            # (RES,RES) bool, sphere silhouette
    rendered = load_exr_rgb(args.exr)

    if rendered.shape[:2] != ref.shape[:2]:
        raise SystemExit(
            f"shape mismatch: rendered {rendered.shape[:2]} vs reference "
            f"{ref.shape[:2]} -- Format in the RIB must match RES in "
            f"render_gt_comparison.py exactly.")

    # Lit mask from the analytic reference itself: inside the sphere AND
    # not a cos(N.L)<=0 dark pixel that render_material() zeroed out.
    # (Pixels outside the sphere are background in BOTH images and are
    # correctly excluded here, not just zero-vs-zero coincidence.)
    lit = mask & (ref.sum(axis=-1) > 0)
    if lit.sum() == 0:
        raise SystemExit(
            "no lit pixels in the reference -- re-check render_gt_comparison.py's "
            "LIGHT_DIR/geometry before trusting anything downstream.")

    diff = rendered[lit] - ref[lit]
    mae_per_channel = np.abs(diff).mean(axis=0)
    mae_total = float(np.abs(diff).mean())
    max_abs = float(np.abs(diff).max())

    rendered_bright = rendered[mask].sum() if mask.any() else 0.0
    print(f"rendered sphere pixels: {int(mask.sum())}, lit-per-reference: {int(lit.sum())}")
    if rendered_bright == 0.0:
        print("WARNING: every sphere pixel in the RENDER is exactly 0 -- almost "
              "certainly the light direction is backward (see a2_distant_alum.rib's "
              "convention caveat: try negating the RotateY angle). Don't trust the "
              "MAE below until this is fixed.")

    print(f"\n{os.path.basename(args.exr)} vs {args.label} analytic NBRDF reference "
          f"(render_gt_comparison.py)")
    print(f"MAE per channel: R={mae_per_channel[0]:.5f} G={mae_per_channel[1]:.5f} "
          f"B={mae_per_channel[2]:.5f}   mean={mae_total:.5f}   max|diff|={max_abs:.5f}")
    print("\nCompare this MAE against the purely-analytic GT-vs-NBRDF MAE that "
          "render_gt_comparison.py itself printed for this material (no renderer "
          "involved). Per the protocol: a MUCH larger gap here points at the plugin's "
          "angle conversion or cosine-term handling, not the weights.")

    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["channel", "mae"])
            for c, name in enumerate("RGB"):
                wr.writerow([name, f"{mae_per_channel[c]:.6f}"])
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
