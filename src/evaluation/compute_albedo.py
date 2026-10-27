"""
White-furnace / energy-conservation reference for the NBRDF.

For a fixed view direction wo, the directional albedo is
    rho(wo) = integral over the hemisphere of  f(wi, wo) * cos(theta_i) dwi
Under a uniform white environment of unit radiance, a pixel whose view
direction is wo shows exactly rho(wo). So this script produces BOTH:

  * the per-view-angle target curve to compare a white-furnace RENDER against
    (Step B of the correctness gate), and
  * an energy-conservation check: a physical BRDF has rho(wo) <= 1 in every
    channel for every wo. Values above 1 mean the network emits more energy
    than it receives -- a real defect to report.

It evaluates rho two ways, with the SAME half/diff math and index mapping as
render_gt_comparison.py and train_nbrdf_v2.ipynb:
  * GT   -- raw MERL table lookup
  * NBRDF -- the trained network

Integration is uniform-hemisphere Monte Carlo (pdf = 1/2pi), so
    rho ~= (2*pi / N) * sum_k  f(wi_k, wo) * cos(theta_i_k).

Usage:
    python compute_albedo.py --merl alum-bronze.binary --pt nbrdf_weights_v2.pt
    python compute_albedo.py --self-test      # no files needed; checks the integrator

Requires: numpy; torch only if --pt is given. (matplotlib optional, for the plot.)
"""
import argparse
import numpy as np

N_THETA_H, N_THETA_D, N_PHI_D = 90, 90, 180
SCALE = np.array([1.0 / 1500.0, 1.15 / 1500.0, 1.66 / 1500.0])


# ---------------------------------------------------------------- MERL
def load_merl_dense(path):
    with open(path, "rb") as f:
        dims = np.fromfile(f, dtype=np.int32, count=3)
        assert tuple(dims) == (N_THETA_H, N_THETA_D, N_PHI_D), \
            f"Got header {tuple(dims)} -- not a MERL .binary file?"
        raw = np.fromfile(f, dtype=np.float64)
    n = N_THETA_H * N_THETA_D * N_PHI_D
    rgb = raw.reshape(3, n).T.reshape(N_THETA_H, N_THETA_D, N_PHI_D, 3) * SCALE
    return rgb.astype(np.float32)


# ---------------------------------------------- half/diff (matches C++ + val script)
def standard_to_half_diff(wi, wo):
    h = wi + wo
    h = h / np.clip(np.linalg.norm(h, axis=-1, keepdims=True), 1e-8, None)
    theta_h = np.arccos(np.clip(h[:, 2], -1.0, 1.0))
    phi_h = np.arctan2(h[:, 1], h[:, 0])

    def rot_z(v, a):
        c, s = np.cos(a), np.sin(a)
        return np.stack([c*v[:, 0] - s*v[:, 1], s*v[:, 0] + c*v[:, 1], v[:, 2]], -1)

    def rot_y(v, a):
        c, s = np.cos(a), np.sin(a)
        return np.stack([c*v[:, 0] + s*v[:, 2], v[:, 1], -s*v[:, 0] + c*v[:, 2]], -1)

    d = rot_y(rot_z(wi, -phi_h), -theta_h)
    theta_d = np.arccos(np.clip(d[:, 2], -1.0, 1.0))
    phi_d = np.arctan2(d[:, 1], d[:, 0])
    phi_d = np.where(phi_d < 0, phi_d + np.pi, phi_d)
    return theta_h, theta_d, phi_d


def merl_eval(merl):
    def f(wi, wo):
        th, td, pd = standard_to_half_diff(wi, wo)
        th_i = np.clip((np.sqrt(np.clip(th, 0, None) / (np.pi/2)) * N_THETA_H).astype(int), 0, N_THETA_H-1)
        td_i = np.clip((td / (np.pi/2) * N_THETA_D).astype(int), 0, N_THETA_D-1)
        pd_i = np.clip((pd / np.pi * N_PHI_D).astype(int), 0, N_PHI_D-1)
        return np.clip(merl[th_i, td_i, pd_i], 0.0, None)
    return f


# ---------------------------------------------------------------- network
def nbrdf_eval(pt_path, hid=64):
    import torch
    import torch.nn as nn

    class NBRDF(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(6, hid), nn.ReLU(),
                                     nn.Linear(hid, hid), nn.ReLU(),
                                     nn.Linear(hid, 3))

        def forward(self, x):
            return torch.exp(self.net(x)) - 1e-3

    model = NBRDF()
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()

    def f(wi, wo):
        th, td, pd = standard_to_half_diff(wi, wo)
        hx, hy, hz = np.sin(th), np.zeros_like(th), np.cos(th)
        dx, dy, dz = np.sin(td)*np.cos(pd), np.sin(td)*np.sin(pd), np.cos(td)
        net_in = np.stack([hx, hy, hz, dx, dy, dz], -1).astype(np.float32)
        with torch.no_grad():
            return np.clip(model(torch.from_numpy(net_in)).numpy(), 0.0, None)
    return f


# ---------------------------------------------------------------- albedo integral
def albedo(eval_fn, theta_o_deg, n_samples=200_000, seed=0):
    """Uniform-hemisphere Monte Carlo estimate of directional albedo for one wo."""
    rng = np.random.default_rng(seed)
    th_o = np.deg2rad(theta_o_deg)
    wo = np.tile(np.array([np.sin(th_o), 0.0, np.cos(th_o)], np.float32), (n_samples, 1))

    # uniform hemisphere sample: z = u1 in (0,1], pdf = 1/(2pi)
    u1 = rng.random(n_samples).astype(np.float32)
    u2 = rng.random(n_samples).astype(np.float32)
    z = u1
    r = np.sqrt(np.clip(1.0 - z*z, 0.0, None))
    phi = 2*np.pi*u2
    wi = np.stack([r*np.cos(phi), r*np.sin(phi), z], -1).astype(np.float32)

    f = eval_fn(wi, wo)                      # (N,3)
    cos_i = wi[:, 2:3]                        # cos(theta_i) = wi.z
    integrand = f * cos_i                     # (N,3)
    return (2*np.pi) * integrand.mean(axis=0)  # (3,) rho per channel


def run(eval_fn, label, angles):
    print(f"\n{label}: directional albedo rho(theta_o) per RGB channel")
    print(f"{'theta_o':>8} | {'R':>8} {'G':>8} {'B':>8} | note")
    worst = 0.0
    rows = []
    for a in angles:
        rho = albedo(eval_fn, a)
        worst = max(worst, float(rho.max()))
        flag = "  <-- >1: energy gain!" if rho.max() > 1.0 else ""
        print(f"{a:8.0f} | {rho[0]:8.4f} {rho[1]:8.4f} {rho[2]:8.4f} |{flag}")
        rows.append((a, *rho))
    print(f"  max channel albedo over all angles: {worst:.4f} "
          f"({'OK, <= 1' if worst <= 1.0 else 'EXCEEDS 1 -- not energy conserving'})")
    return rows


def self_test():
    """A Lambertian BRDF f = a/pi (constant) has rho = a exactly.
    Checks the Monte Carlo integrator independent of any file."""
    for a in (0.2, 0.5, 0.9):
        const = (a / np.pi)

        def f(wi, wo, c=const):
            return np.full((wi.shape[0], 3), c, np.float32)

        rho = albedo(f, theta_o_deg=30.0, n_samples=400_000)
        err = abs(float(rho[0]) - a)
        status = "OK" if err < 5e-3 else "FAIL"
        print(f"Lambertian a={a:.2f}: rho={rho[0]:.4f}  |err|={err:.4f}  [{status}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merl", help="path to MERL .binary (ground-truth albedo)")
    ap.add_argument("--pt", help="path to trained .pt (network albedo)")
    ap.add_argument("--self-test", action="store_true", help="check the integrator, no files")
    ap.add_argument("--csv", help="optional path to write a CSV of the curves")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    angles = [0, 15, 30, 45, 60, 70, 80, 85]
    all_rows = {}
    if args.merl:
        all_rows["GT"] = run(merl_eval(load_merl_dense(args.merl)), "GT (MERL table)", angles)
    if args.pt:
        all_rows["NBRDF"] = run(nbrdf_eval(args.pt), "NBRDF (trained network)", angles)
    if not all_rows:
        ap.error("give --merl and/or --pt, or --self-test")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["source", "theta_o_deg", "rhoR", "rhoG", "rhoB"])
            for src, rows in all_rows.items():
                for a, r, g, b in rows:
                    wr.writerow([src, a, f"{r:.6f}", f"{g:.6f}", f"{b:.6f}"])
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
