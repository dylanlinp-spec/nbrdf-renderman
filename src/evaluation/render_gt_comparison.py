"""
Reproduces the paper's "GT DirL" / "NBRDF DirL" comparison style:
an orthographic sphere lit by a single directional light, evaluated
ANALYTICALLY per-pixel (no path tracing, no Monte Carlo noise, no
exposure tuning headaches) against both the raw MERL table (ground
truth) and your trained NBRDF network, side by side.

Uses the exact same Rusinkiewicz half/diff angle math as NBrdf_v2.cpp
(StandardToHalfDiff) and the exact same network architecture as the
training notebook, so results are consistent with both.

Requires: numpy, torch, opencv-python (same env you trained in).

Run (from inside the render folder):
    python path/to/render_gt_comparison.py
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import torch
import torch.nn as nn
import cv2

# -----------------------------------------------------------------
# Materials to render: (label, MERL .binary path, trained .pt path)
#
# Repository edit (paths only, no change to the computation): the original
# file hard-coded the author's own Windows folders. The defaults below point
# at this repository's layout instead, and each one can be overridden with
# an environment variable:
#   MERL_DIR        folder holding alum-bronze.binary and chrome.binary
#                   (default: <repo>/data/merl)
#   WEIGHTS_DIR     folder holding the trained .pt files
#                   (default: <repo>/results/weights)
#   ALUM_BRONZE_PT  file name of the alum-bronze .pt inside WEIGHTS_DIR
#                   (default: nbrdf_weights_v2.pt, the file the original
#                   results used -- see doc/known_issues.md)
# The outputs (gt_vs_nbrdf.png and the two .npy files per material) are
# written to the current working directory, exactly as before, because the
# author always ran this script from inside the render folder.
# -----------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MERL_DIR = os.environ.get("MERL_DIR", os.path.join(REPO_ROOT, "data", "merl"))
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", os.path.join(REPO_ROOT, "results", "weights"))
ALUM_BRONZE_PT = os.environ.get("ALUM_BRONZE_PT", "nbrdf_weights_v2.pt")

MATERIALS = [
    ("alum-bronze", os.path.join(MERL_DIR, "alum-bronze.binary"),
     os.path.join(WEIGHTS_DIR, ALUM_BRONZE_PT)),
    ("chrome", os.path.join(MERL_DIR, "chrome.binary"),
     os.path.join(WEIGHTS_DIR, "nbrdf_weights_v2_chrome.pt")),
]

OUT_PATH = "gt_vs_nbrdf.png"

RES = 384              # per-panel resolution
HID_DIM = 64
EXPOSURE = 1.5          # tune this if everything looks too dark/bright
GAMMA = 1.0 / 2.2

# Directional light, fixed in world space (view is always +Z, orthographic).
LIGHT_DIR = np.array([0.35, 0.55, 0.75])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
VIEW_DIR = np.array([0.0, 0.0, 1.0])

N_THETA_H, N_THETA_D, N_PHI_D = 90, 90, 180
SCALE = np.array([1.0 / 1500.0, 1.15 / 1500.0, 1.66 / 1500.0])


# -----------------------------------------------------------------
# MERL loading (dense grid, matches train_nbrdf_v2.ipynb's load_merl)
# -----------------------------------------------------------------
def load_merl_dense(path):
    with open(path, "rb") as f:
        dims = np.fromfile(f, dtype=np.int32, count=3)
        assert tuple(dims) == (N_THETA_H, N_THETA_D, N_PHI_D), \
            f"Got header {tuple(dims)} -- not a MERL .binary file?"
        raw = np.fromfile(f, dtype=np.float64)
    n = N_THETA_H * N_THETA_D * N_PHI_D
    rgb = raw.reshape(3, n).T.reshape(N_THETA_H, N_THETA_D, N_PHI_D, 3) * SCALE
    return rgb.astype(np.float32)  # (90, 90, 180, 3), may contain small negatives -> clip later


# -----------------------------------------------------------------
# NBRDF network (matches the notebook's NBRDF class exactly)
# -----------------------------------------------------------------
class NBRDF(nn.Module):
    def __init__(self, hid=HID_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 3),
        )

    def forward(self, x):
        return torch.exp(self.net(x)) - 1e-3


# -----------------------------------------------------------------
# Rusinkiewicz half/diff angles, vectorized -- same math as
# StandardToHalfDiff() in NBrdf_v2.cpp, just batched over pixels.
# wi, wo are (N,3) arrays in each pixel's LOCAL shading frame
# (normal = local +Z).
# -----------------------------------------------------------------
def standard_to_half_diff(wi, wo):
    h = wi + wo
    h = h / np.clip(np.linalg.norm(h, axis=-1, keepdims=True), 1e-8, None)
    theta_h = np.arccos(np.clip(h[:, 2], -1.0, 1.0))
    phi_h = np.arctan2(h[:, 1], h[:, 0])

    def rotate_z(v, a):
        c, s = np.cos(a), np.sin(a)
        return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], -1)

    def rotate_y(v, a):
        c, s = np.cos(a), np.sin(a)
        return np.stack([c * v[:, 0] + s * v[:, 2], v[:, 1], -s * v[:, 0] + c * v[:, 2]], -1)

    d = rotate_y(rotate_z(wi, -phi_h), -theta_h)
    theta_d = np.arccos(np.clip(d[:, 2], -1.0, 1.0))
    phi_d = np.arctan2(d[:, 1], d[:, 0])
    phi_d = np.where(phi_d < 0, phi_d + np.pi, phi_d)
    return theta_h, theta_d, phi_d


def build_sphere_pixels(res):
    """Orthographic unit sphere. Returns per-pixel world normal N (K,3),
    boolean mask of which (row,col) are on the sphere, and the mask
    shaped (res,res)."""
    xs = np.linspace(-1, 1, res)
    ys = np.linspace(1, -1, res)  # image row 0 = top = +y
    X, Y = np.meshgrid(xs, ys)
    r2 = X ** 2 + Y ** 2
    mask = r2 <= 1.0
    Z = np.zeros_like(X)
    Z[mask] = np.sqrt(1.0 - r2[mask])
    N = np.stack([X[mask], Y[mask], Z[mask]], axis=-1)
    return N, mask


def local_frames(N):
    """Per-normal orthonormal (T,B,N) basis. Isotropic BRDF math is
    invariant to the choice of T around N, so any consistent basis works."""
    up = np.tile(np.array([0.0, 1.0, 0.0]), (N.shape[0], 1))
    degenerate = np.abs(N[:, 1]) > 0.99
    up[degenerate] = np.array([1.0, 0.0, 0.0])
    T = np.cross(up, N)
    T = T / np.clip(np.linalg.norm(T, axis=-1, keepdims=True), 1e-8, None)
    B = np.cross(N, T)
    return T, B


def to_local(world_dir, T, B, N):
    world_dir = np.tile(world_dir, (N.shape[0], 1))
    return np.stack([
        np.sum(world_dir * T, axis=-1),
        np.sum(world_dir * B, axis=-1),
        np.sum(world_dir * N, axis=-1),
    ], axis=-1)


def render_material(label, merl_path, pt_path, res):
    print(f"[{label}] building geometry...")
    N, mask = build_sphere_pixels(res)
    T, B = local_frames(N)
    wi_local = to_local(LIGHT_DIR, T, B, N)
    wo_local = to_local(VIEW_DIR, T, B, N)

    # cos(N.L) and cos(N.V) in world space == wi_local.z / wo_local.z
    # in the local frame (N is local +Z by construction).
    cos_nl = wi_local[:, 2]
    cos_nv = wo_local[:, 2]
    lit = (cos_nl > 0) & (cos_nv > 0)

    theta_h, theta_d, phi_d = standard_to_half_diff(wi_local, wo_local)

    # ---- Ground truth: direct MERL table lookup ----
    print(f"[{label}] loading MERL table + looking up GT...")
    merl = load_merl_dense(merl_path)
    th_i = np.clip((np.sqrt(np.clip(theta_h, 0, None) / (np.pi / 2)) * N_THETA_H).astype(int), 0, N_THETA_H - 1)
    td_i = np.clip((theta_d / (np.pi / 2) * N_THETA_D).astype(int), 0, N_THETA_D - 1)
    pd_i = np.clip((phi_d / np.pi * N_PHI_D).astype(int), 0, N_PHI_D - 1)
    gt_rgb = merl[th_i, td_i, pd_i]
    gt_rgb = np.clip(gt_rgb, 0.0, None) * cos_nl[:, None]
    gt_rgb[~lit] = 0.0

    # ---- NBRDF: trained network forward pass ----
    print(f"[{label}] running trained network...")
    model = NBRDF(HID_DIM)
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()

    hx, hy, hz = np.sin(theta_h), np.zeros_like(theta_h), np.cos(theta_h)
    dx = np.sin(theta_d) * np.cos(phi_d)
    dy = np.sin(theta_d) * np.sin(phi_d)
    dz = np.cos(theta_d)
    net_in = np.stack([hx, hy, hz, dx, dy, dz], axis=-1).astype(np.float32)
    with torch.no_grad():
        net_rgb = model(torch.from_numpy(net_in)).numpy()
    net_rgb = np.clip(net_rgb, 0.0, None) * cos_nl[:, None]
    net_rgb[~lit] = 0.0

    mae = float(np.abs(gt_rgb[lit] - net_rgb[lit]).mean()) if lit.any() else float("nan")
    print(f"[{label}] mean abs error (GT vs NBRDF, lit pixels): {mae:.5f}")

    # ---- Scatter both back into (res,res,3) images, black background ----
    def to_image(flat_rgb):
        img = np.zeros((res, res, 3), dtype=np.float32)
        img[mask] = flat_rgb
        return img

    return to_image(gt_rgb), to_image(net_rgb), mae


def tonemap(img):
    exposed = img * EXPOSURE
    return (np.clip(exposed, 0.0, 1.0) ** GAMMA * 255.0).astype(np.uint8)


LABEL_BAND = 40
FONT = cv2.FONT_HERSHEY_SIMPLEX

rows = []
for label, merl_path, pt_path in MATERIALS:
    gt_img, net_img, mae = render_material(label, merl_path, pt_path, RES)

    # A2 (performance-evaluation-protocol.md): dump the RAW LINEAR
    # (un-tonemapped, un-exposure-scaled) NBRDF image + its sphere mask
    # so a real RenderMan render of the matching distant-light scene
    # (a2_distant_<material>.rib) can be diffed against this exact
    # per-pixel reference. The PNG below is tonemapped for viewing only
    # -- diffing against it would bake in EXPOSURE/GAMMA and give a
    # meaningless MAE.
    _, mask_img = build_sphere_pixels(RES)
    np.save(f"{label}_nbrdf_linear.npy", net_img)
    np.save(f"{label}_mask.npy", mask_img)
    print(f"[{label}] wrote {label}_nbrdf_linear.npy + {label}_mask.npy (raw linear A2 reference)")

    # BGR for cv2
    gt_bgr = tonemap(gt_img[..., ::-1])
    net_bgr = tonemap(net_img[..., ::-1])
    gap = np.full((RES, 12, 3), 30, dtype=np.uint8)
    panels = np.concatenate([gt_bgr, gap, net_bgr], axis=1)

    band = np.full((LABEL_BAND, panels.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(band, f"{label}  |  GT", (10, 27), FONT, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(band, f"NBRDF  (MAE {mae:.4f})", (RES + 22, 27), FONT, 0.6, (230, 230, 230), 1, cv2.LINE_AA)

    row = np.concatenate([band, panels], axis=0)
    rows.append(row)
    print(f"[{label}] done.")

row_gap = np.full((12, rows[0].shape[1], 3), 30, dtype=np.uint8)
composite_rows = []
for i, r in enumerate(rows):
    if i > 0:
        composite_rows.append(row_gap)
    composite_rows.append(r)
composite = np.concatenate(composite_rows, axis=0)

cv2.imwrite(OUT_PATH, composite)
print(f"\nWrote {OUT_PATH}")
print("Layout: each row is one material, GT (raw MERL) on the left, NBRDF (trained network) on the right.")
print("If it looks too dark/bright, adjust EXPOSURE near the top and re-run.")
