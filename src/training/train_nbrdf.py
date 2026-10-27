"""
train_nbrdf.py -- train one neural BRDF (NBRDF) on one MERL material and
export it for the RenderMan plugin.

This is a line-for-line command-line version of the notebook that produced
every network used in this project (train_nbrdf_v2.ipynb for alum-bronze,
train_nbrdf_chrome.ipynb for chrome; the two notebooks differ only in their
file paths). The loader, network, loss, sampling, optimizer, learning-rate
schedule, step count, batch size, seeds and export layout are unchanged.

The only differences from the notebooks:
  * file paths come from command-line arguments instead of being typed into
    a cell;
  * the two plots are saved as PNG files instead of shown in a window;
  * an OPTIONAL --torch-seed flag. The original notebooks never seeded
    PyTorch, so the network's starting weights were random on every run. By
    default this script does the same (no seed), which means a fresh run
    produces a network that is similar but NOT bit-identical to the shipped
    weights in results/weights/. Pass --torch-seed 0 if you want your own
    runs to be repeatable from now on.

Usage (example values):
    python src/training/train_nbrdf.py \
        --material data/merl/alum-bronze.binary \
        --weights-out results/my_run/alum-bronze.pt \
        --renderman-out results/my_run/alum-bronze.bin

Outputs:
    <weights-out>          PyTorch state dict (used by the Python evaluation scripts)
    <renderman-out>        flat float32 file the C++ plugin loads (19,212 bytes)
    <plot-dir>/<stem>_training_curve.png
    <plot-dir>/<stem>_comparison.png

Expected runtime: about 5-20 minutes on a laptop CPU (20,000 steps).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # save figures to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

HID_DIM = 64   # was 21 in v1

# -----------------------------------------------------------------
# Step 1 -- MERL loader (fixed constants for every MERL material)
# -----------------------------------------------------------------
N_THETA_H, N_THETA_D, N_PHI_D = 90, 90, 180
SCALE = np.array([1.0/1500.0, 1.15/1500.0, 1.66/1500.0])


def load_merl(path):
    with open(path, "rb") as f:
        dims = np.fromfile(f, dtype=np.int32, count=3)
        assert tuple(dims) == (N_THETA_H, N_THETA_D, N_PHI_D), \
            f"Got header {tuple(dims)} -- not a MERL .binary file?"
        raw = np.fromfile(f, dtype=np.float64)

    n = N_THETA_H * N_THETA_D * N_PHI_D
    rgb = raw.reshape(3, n).T.reshape(N_THETA_H, N_THETA_D, N_PHI_D, 3) * SCALE

    ih, idd, ip = np.meshgrid(
        np.arange(N_THETA_H), np.arange(N_THETA_D), np.arange(N_PHI_D), indexing="ij"
    )
    theta_h = (ih / N_THETA_H) ** 2 * (np.pi / 2)
    theta_d = (idd / N_THETA_D) * (np.pi / 2)
    phi_d   = (ip / N_PHI_D) * np.pi

    table = np.stack([theta_h, theta_d, phi_d,
                      rgb[...,0], rgb[...,1], rgb[...,2]], -1).reshape(-1, 6)
    valid = table[:, 3:].min(axis=1) >= 0
    return table[valid].astype(np.float32)


def angles_to_input(theta_h, theta_d, phi_d):
    hx, hy, hz = np.sin(theta_h), np.zeros_like(theta_h), np.cos(theta_h)
    dx = np.sin(theta_d) * np.cos(phi_d)
    dy = np.sin(theta_d) * np.sin(phi_d)
    dz = np.cos(theta_d)
    return np.stack([hx, hy, hz, dx, dy, dz], axis=-1)


# -----------------------------------------------------------------
# Step 2 -- the network (6 -> 64 -> 64 -> 3)
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
# Step 3 -- train (20K steps, cosine LR schedule)
# -----------------------------------------------------------------
def train(material_path, steps=20000, batch_size=2048, lr=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    table = load_merl(material_path)
    print(f"Loaded {len(table):,} valid measurements from {material_path}")

    perm = rng.permutation(len(table))
    n_val = len(table) // 10
    val_table, train_table = table[perm[:n_val]], table[perm[n_val:]]

    # Brightness-weighted adaptive sampling -- oversamples specular peaks
    brightness = train_table[:, 3:].mean(axis=1)
    weights = brightness + brightness.mean() * 0.1
    probs = weights / weights.sum()

    model = NBRDF()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    losses = []
    for step in range(steps):
        idx = rng.choice(len(train_table), size=batch_size, p=probs)
        batch = train_table[idx]
        x = torch.tensor(angles_to_input(*batch[:, :3].T))
        y = torch.tensor(batch[:, 3:])

        pred = model(x)
        loss = ((torch.log(pred + 1e-3) - torch.log(y + 1e-3)) ** 2).mean()

        opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()
        losses.append(loss.item())
        if step % 2000 == 0 or step == steps - 1:
            print(f"  step {step:5d}   train log-MSE = {loss.item():.4f}   lr = {scheduler.get_last_lr()[0]:.2e}")

    with torch.no_grad():
        x_val = torch.tensor(angles_to_input(*val_table[:, :3].T))
        pred_val = model(x_val).numpy()
    mae = np.abs(pred_val - val_table[:, 3:]).mean()
    print(f"\nValidation MAE: {mae:.5f}")
    return model, table, losses


# -----------------------------------------------------------------
# Step 4 -- training curve
# -----------------------------------------------------------------
def plot_training_curve(losses, out_path):
    plt.figure(figsize=(6, 3))
    plt.plot(np.convolve(losses, np.ones(100)/100, 'valid'))
    plt.xlabel('Step'); plt.ylabel('Log-MSE (smoothed)')
    plt.title('v2 Training Curve (HID=64, 20K steps)')
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()
    print(f"Saved training curve to {out_path}")


# -----------------------------------------------------------------
# Step 5 -- sanity-check plot
# -----------------------------------------------------------------
def plot_slice_comparison(model, table, out_path, phi_d_slice=0.0, res=90):
    theta_h_vals = np.linspace(0, np.pi/2, res)
    theta_d_vals = np.linspace(0, np.pi/2, res)
    TH, TD = np.meshgrid(theta_h_vals, theta_d_vals, indexing="ij")
    PD = np.full_like(TH, phi_d_slice)

    net_input = angles_to_input(TH, TD, PD)
    with torch.no_grad():
        net_rgb = model(torch.tensor(net_input.astype(np.float32))).numpy()

    grid = np.zeros((N_THETA_H, N_THETA_D, N_PHI_D, 3))
    th_i = np.clip((np.sqrt(table[:,0]/(np.pi/2))*N_THETA_H).astype(int), 0, N_THETA_H-1)
    td_i = np.clip((table[:,1]/(np.pi/2)*N_THETA_D).astype(int), 0, N_THETA_D-1)
    pd_i = np.clip((table[:,2]/np.pi*N_PHI_D).astype(int), 0, N_PHI_D-1)
    grid[th_i, td_i, pd_i] = table[:, 3:]

    ih = np.clip((np.sqrt(TH/(np.pi/2))*N_THETA_H).astype(int), 0, N_THETA_H-1)
    idd = np.clip((TD/(np.pi/2)*N_THETA_D).astype(int), 0, N_THETA_D-1)
    ip = np.clip((PD/np.pi*N_PHI_D).astype(int), 0, N_PHI_D-1)
    raw_rgb = grid[ih, idd, ip]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, img, title in [(axes[0], raw_rgb, "Raw MERL table"),
                           (axes[1], net_rgb, "Trained network (v2, 64-wide)")]:
        ax.imshow(np.clip(img, 0, 1) ** (1/2.2), origin="lower",
                  extent=[0, 90, 0, 90])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("theta_d (deg)"); ax.set_ylabel("theta_h (deg)")
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close(fig)
    print(f"Saved comparison plot to {out_path}")

    print(f"Mean abs error on this slice: {np.abs(raw_rgb - net_rgb).mean():.5f}")


# -----------------------------------------------------------------
# Step 6 -- export weights for the RenderMan Bxdf plugin
#
# Flat float32 file, fixed order:
#   W1 (6x64), b1 (64), W2 (64x64), b2 (64), W3 (64x3), b3 (3)
# Total: 4,803 floats = 19,212 bytes. (The original notebook's markdown
# cell said "4,867 floats = 19,468 bytes"; that comment was wrong, the code
# and the assert below were always right.)
# -----------------------------------------------------------------
def export_weights_for_renderman(model, path):
    sd = model.state_dict()
    # PyTorch Linear stores weight as (out_features, in_features);
    # write it transposed to (in_features, out_features) so the C++ side
    # can do a straightforward row-major dot product.
    arrays = [
        sd["net.0.weight"].detach().numpy().T,  # (6, HID)
        sd["net.0.bias"].detach().numpy(),       # (HID,)
        sd["net.2.weight"].detach().numpy().T,  # (HID, HID)
        sd["net.2.bias"].detach().numpy(),       # (HID,)
        sd["net.4.weight"].detach().numpy().T,  # (HID, 3)
        sd["net.4.bias"].detach().numpy(),       # (3,)
    ]
    flat = np.concatenate([a.flatten() for a in arrays]).astype(np.float32)

    expected = 6*HID_DIM + HID_DIM + HID_DIM*HID_DIM + HID_DIM + HID_DIM*3 + 3
    assert flat.size == expected, \
        f"Weight count mismatch: got {flat.size}, expected {expected}"

    flat.tofile(path)
    print(f"Exported {flat.size} floats ({flat.nbytes} bytes) to {path}")
    print(f"Expected by C++ plugin: {expected} floats ({expected*4} bytes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--material", required=True,
                    help="path to a MERL .binary file, e.g. data/merl/alum-bronze.binary")
    ap.add_argument("--weights-out", required=True,
                    help="where to save the PyTorch weights (.pt)")
    ap.add_argument("--renderman-out", required=True,
                    help="where to save the flat float32 weights for the plugin (.bin)")
    ap.add_argument("--plot-dir", default=None,
                    help="folder for the two PNG plots (default: same folder as --weights-out)")
    ap.add_argument("--torch-seed", type=int, default=None,
                    help="OPTIONAL. Seed PyTorch's weight initialization. The original "
                         "runs did not, so leave this out to match them.")
    args = ap.parse_args()

    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
        print(f"PyTorch seeded with {args.torch_seed} (the original runs were unseeded)")

    for out in (args.weights_out, args.renderman_out):
        folder = os.path.dirname(os.path.abspath(out))
        os.makedirs(folder, exist_ok=True)
    plot_dir = args.plot_dir or os.path.dirname(os.path.abspath(args.weights_out))
    os.makedirs(plot_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.weights_out))[0]

    model, table, losses = train(args.material)
    torch.save(model.state_dict(), args.weights_out)
    print(f"Saved PyTorch weights to {args.weights_out}")

    plot_training_curve(losses, os.path.join(plot_dir, f"{stem}_training_curve.png"))
    plot_slice_comparison(model, table, os.path.join(plot_dir, f"{stem}_comparison.png"))
    export_weights_for_renderman(model, args.renderman_out)


if __name__ == "__main__":
    main()
