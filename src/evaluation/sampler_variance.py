"""
Track B (performance-evaluation-protocol.md): sampler-efficiency, computed
WITHOUT the renderer.

Why: the samplingMode ablation through RenderMan (track_b_noise.py) showed
mixture/cosine-only/GGX-only are bit-identical in the furnace scene -- not
a plugin bug, but evidence (Sec 5.x.3b) that GenerateSample's contribution
never reaches the image for PxrDomeLight. That code path is dead in the
renderer right now, so Track B can't be measured there until the dome-
propagation defect (Sec 3.4) is fixed.

But the thing Track B actually wants to quantify -- how much noise each
importance sampler produces per sample -- is a property of the trained
network and the three pdf formulas alone. It doesn't need a renderer at
all. For a one-sample Monte Carlo estimator Y = f(wi,wo)*cos(theta_i) / p(wi)
with wi ~ p:

    E[Y]   = integral f*cos dwi                      = rho(theta_o)   (true, independent of p)
    E[Y^2] = integral p(wi) * [f*cos/p(wi)]^2 dwi
           = integral [f*cos]^2 / p(wi) dwi
    Var[Y] = E[Y^2] - rho(theta_o)^2

E[Y^2] is computed here by direct numerical quadrature over the upper
hemisphere (same fine theta/phi grid for all three techniques -- this is
NOT Monte Carlo sampling, it's exact numerical integration of a smooth
integrand, so there's no sampling noise in the number itself). This uses
the identical EvalNetwork/StandardToHalfDiff math and the identical
CosinePDF/GGX_PDF/MixturePDF formulas as NBrdf_v3_diag.cpp -- same
GGX_ALPHA, same near-grazing denominator clamp -- so the result is a
faithful stand-in for what the plugin's samplers would produce, decoupled
from the renderer's broken light-transport path.

Usage:
    python sampler_variance.py --weights nbrdf_weights.bin --label alum-bronze
    python sampler_variance.py --weights nbrdf_weights.bin --label alum-bronze --csv table3_variance.csv
"""
import argparse
import struct
import numpy as np

IN_DIM, HID_DIM, OUT_DIM = 6, 64, 3
TOTAL_WEIGHTS = IN_DIM*HID_DIM + HID_DIM + HID_DIM*HID_DIM + HID_DIM + HID_DIM*OUT_DIM + OUT_DIM
GGX_ALPHA_DEFAULT = 0.05  # matches the shipped NBrdf_v3_diag.cpp; override with --alpha


def load_weights(path):
    with open(path, "rb") as f:
        buf = f.read()
    n = len(buf) // 4
    if n != TOTAL_WEIGHTS:
        raise SystemExit(f"{path}: expected {TOTAL_WEIGHTS} float32 ({TOTAL_WEIGHTS*4} bytes), "
                          f"got {n} ({len(buf)} bytes)")
    flat = np.array(struct.unpack(f"<{n}f", buf), dtype=np.float64)
    i = 0
    W1 = flat[i:i+IN_DIM*HID_DIM].reshape(IN_DIM, HID_DIM); i += IN_DIM*HID_DIM
    b1 = flat[i:i+HID_DIM]; i += HID_DIM
    W2 = flat[i:i+HID_DIM*HID_DIM].reshape(HID_DIM, HID_DIM); i += HID_DIM*HID_DIM
    b2 = flat[i:i+HID_DIM]; i += HID_DIM
    W3 = flat[i:i+HID_DIM*OUT_DIM].reshape(HID_DIM, OUT_DIM); i += HID_DIM*OUT_DIM
    b3 = flat[i:i+OUT_DIM]; i += OUT_DIM
    return W1, b1, W2, b2, W3, b3


def eval_network_batch(w, inp):
    """inp: (N,6) -> (N,3). Same architecture/activation as EvalNetwork in NBrdf_v3_diag.cpp."""
    W1, b1, W2, b2, W3, b3 = w
    h1 = np.maximum(0.0, inp @ W1 + b1)
    h2 = np.maximum(0.0, h1 @ W2 + b2)
    acc = h2 @ W3 + b3
    return np.maximum(0.0, np.exp(acc) - 1e-3)


def standard_to_halfdiff_batch(wi, wo):
    """wi, wo: (N,3) local-frame unit vectors. Returns theta_h, theta_d, phi_d: (N,)."""
    h = wi + wo
    h = h / np.linalg.norm(h, axis=-1, keepdims=True)
    theta_h = np.arccos(np.clip(h[:, 2], -1.0, 1.0))
    phi_h = np.arctan2(h[:, 1], h[:, 0])

    def rotate_z(v, a):
        c, s = np.cos(a), np.sin(a)
        return np.stack([c*v[:, 0] - s*v[:, 1], s*v[:, 0] + c*v[:, 1], v[:, 2]], axis=-1)

    def rotate_y(v, a):
        c, s = np.cos(a), np.sin(a)
        return np.stack([c*v[:, 0] + s*v[:, 2], v[:, 1], -s*v[:, 0] + c*v[:, 2]], axis=-1)

    d = rotate_y(rotate_z(wi, -phi_h), -theta_h)
    theta_d = np.arccos(np.clip(d[:, 2], -1.0, 1.0))
    phi_d = np.arctan2(d[:, 1], d[:, 0])
    phi_d = np.where(phi_d < 0.0, phi_d + np.pi, phi_d)
    return theta_h, theta_d, phi_d


def eval_reflectance_batch(w, wi, wo):
    theta_h, theta_d, phi_d = standard_to_halfdiff_batch(wi, wo)
    h = np.stack([np.sin(theta_h), np.zeros_like(theta_h), np.cos(theta_h)], axis=-1)
    d = np.stack([np.sin(theta_d)*np.cos(phi_d), np.sin(theta_d)*np.sin(phi_d), np.cos(theta_d)],
                 axis=-1)
    inp = np.concatenate([h, d], axis=-1)  # (N,6)
    return eval_network_batch(w, inp)  # (N,3)


def ggx_d(cos_theta, alpha):
    a2 = alpha*alpha
    cos2 = cos_theta*cos_theta
    denom = cos2*(a2 - 1.0) + 1.0
    return a2 / (np.pi * denom * denom)


def cosine_pdf(wi):
    return np.maximum(0.0, wi[:, 2]) / np.pi


def ggx_pdf(wi, wo, alpha):
    h = wi + wo
    h = h / np.linalg.norm(h, axis=-1, keepdims=True)
    cos_theta_h = np.maximum(0.0, h[:, 2])
    D = ggx_d(cos_theta_h, alpha)
    wo_dot_h = np.maximum(np.abs(np.sum(wo * h, axis=-1)), 1e-3)  # v3's clamp
    return D * cos_theta_h / (4.0 * wo_dot_h)


def eval_reflectance_cached(w, wi, wo_b):
    """Cache EvalReflectance(wi,wo) per theta_o -- it doesn't depend on alpha,
    so the sweep doesn't need to re-run the (slow) network for every alpha."""
    return eval_reflectance_batch(w, wi, wo_b)


def mixture_pdf(wi, wo, alpha):
    return 0.5*cosine_pdf(wi) + 0.5*ggx_pdf(wi, wo, alpha)


def hemisphere_grid(n_theta=400, n_phi=800):
    """Returns wi directions (N,3), solid-angle weights (N,) for a fine
    theta/phi quadrature grid over the upper hemisphere (midpoint rule)."""
    dtheta = (np.pi/2) / n_theta
    dphi = (2*np.pi) / n_phi
    theta_c = (np.arange(n_theta) + 0.5) * dtheta
    phi_c = (np.arange(n_phi) + 0.5) * dphi
    TH, PH = np.meshgrid(theta_c, phi_c, indexing="ij")
    sin_t, cos_t = np.sin(TH), np.cos(TH)
    wi = np.stack([sin_t*np.cos(PH), sin_t*np.sin(PH), cos_t], axis=-1).reshape(-1, 3)
    domega = (sin_t * dtheta * dphi).reshape(-1)
    return wi, domega


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="raw float32 .bin, same format NBrdf_v3_diag.cpp loads")
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-theta", type=int, default=400)
    ap.add_argument("--n-phi", type=int, default=800)
    ap.add_argument("--alphas", type=float, nargs="+", default=[GGX_ALPHA_DEFAULT],
                     help="GGX_ALPHA value(s) to sweep, e.g. --alphas 0.02 0.05 0.1 0.2 "
                          "(default: just the shipped 0.05)")
    ap.add_argument("--csv", help="optional CSV output")
    args = ap.parse_args()

    w = load_weights(args.weights)
    wi, domega = hemisphere_grid(args.n_theta, args.n_phi)
    cos_i = wi[:, 2]

    theta_o_list = [5.6, 16.9, 28.1, 39.4, 50.6, 61.9, 73.1, 84.4]  # match furnace_check.py bin centers
    rows = []

    # f(wi,wo) and rho don't depend on alpha -- compute once per theta_o, reuse across the sweep.
    cache = {}
    for theta_o_deg in theta_o_list:
        theta_o = np.radians(theta_o_deg)
        wo = np.array([np.sin(theta_o), 0.0, np.cos(theta_o)])
        wo_b = np.broadcast_to(wo, wi.shape)
        f = eval_reflectance_batch(w, wi, wo_b)
        fc = f * cos_i[:, None]
        rho = np.sum(fc * domega[:, None], axis=0)
        p_cos = cosine_pdf(wi)
        cache[theta_o_deg] = (wo_b, fc, rho, p_cos)

    def variance(fc, rho, domega, p):
        p_safe = np.maximum(p, 1e-12)
        integrand = (fc**2) / p_safe[:, None]
        e_y2 = np.sum(integrand * domega[:, None], axis=0)
        return e_y2 - rho**2

    for alpha in args.alphas:
        print(f"\n{args.label}: single-sample estimator variance per technique "
              f"(exact quadrature, {args.n_theta}x{args.n_phi} grid, GGX_ALPHA={alpha})")
        print(f"{'theta_o':>8} | {'rho (mean)':>11} | {'Var cosine':>11} {'Var GGX':>11} {'Var mixture':>12} "
              f"| {'GGX/mix':>8} {'cos/mix':>8}")
        for theta_o_deg in theta_o_list:
            wo_b, fc, rho, p_cos = cache[theta_o_deg]
            p_ggx = ggx_pdf(wi, wo_b, alpha)
            p_mix = 0.5 * p_cos + 0.5 * p_ggx

            var_cos = variance(fc, rho, domega, p_cos)
            var_ggx = variance(fc, rho, domega, p_ggx)
            var_mix = variance(fc, rho, domega, p_mix)

            vc, vg, vm = var_cos.mean(), var_ggx.mean(), var_mix.mean()
            ratio_ggx = vg / vm if vm > 0 else float("nan")
            ratio_cos = vc / vm if vm > 0 else float("nan")
            print(f"{theta_o_deg:8.1f} | {rho.mean():11.5f} | {vc:11.6f} {vg:11.6f} {vm:12.6f} "
                  f"| {ratio_ggx:8.2f} {ratio_cos:8.2f}")
            rows.append((alpha, theta_o_deg, rho, var_cos, var_ggx, var_mix))

    print("\nGGX/mix and cos/mix columns: how many times noisier (>1) or quieter (<1) that single "
          "technique is relative to the shipped 50/50 mixture, at that view angle, per one sample. "
          "This is Sec 3.2's anti-correlated-proxy claim made numeric, independent of the renderer.")
    if len(args.alphas) > 1:
        print("\nAcross alpha: as GGX_ALPHA widens (narrow -> broad lobe), expect GGX-only's variance "
              "to fall toward cosine-only's (a wide-enough GGX lobe starts resembling a broad cosine "
              "lobe) -- watch whether the mixture's advantage shrinks at the widest alpha tested.")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["alpha", "theta_o", "rho_R", "rho_G", "rho_B",
                         "var_cosine_R", "var_cosine_G", "var_cosine_B",
                         "var_ggx_R", "var_ggx_G", "var_ggx_B",
                         "var_mixture_R", "var_mixture_G", "var_mixture_B"])
            for alpha, theta_o_deg, rho, var_cos, var_ggx, var_mix in rows:
                wr.writerow([alpha, theta_o_deg, *rho, *var_cos, *var_ggx, *var_mix])
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
