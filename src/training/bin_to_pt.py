"""
bin_to_pt.py -- turn a plugin weights file (.bin) back into a PyTorch .pt
file, so the Python evaluation scripts can analyse EXACTLY the network the
renderer used.

Why this exists: the Python scripts (compute_albedo.py,
render_gt_comparison.py) load .pt files, while the RenderMan plugin loads
.bin files. For chrome the shipped .pt and .bin hold the same network. For
alum-bronze they do not: the .pt file was overwritten by a later retraining
run after the .bin had already been exported, so the .pt describes a
slightly different network than the one in every render. See
doc/known_issues.md. This script rebuilds a .pt from the .bin so the two
always match.

It does not change any weight. It reads the 4,803 float32 values in the
order the plugin reads them (W1, b1, W2, b2, W3, b3), undoes the transpose
the exporter applied, and saves them under the same names PyTorch uses.

Usage:
    python src/training/bin_to_pt.py \
        --bin results/weights/nbrdf_weights.bin \
        --out results/weights/nbrdf_weights_from_bin.pt

Check it worked: the script re-exports the .pt it just wrote and confirms
the result is byte-for-byte identical to the input .bin.
"""
import argparse

import numpy as np
import torch

IN_DIM, HID_DIM, OUT_DIM = 6, 64, 3
LAYOUT = [  # (state-dict name, shape as stored in the .bin)
    ("net.0.weight", (IN_DIM, HID_DIM)),
    ("net.0.bias", (HID_DIM,)),
    ("net.2.weight", (HID_DIM, HID_DIM)),
    ("net.2.bias", (HID_DIM,)),
    ("net.4.weight", (HID_DIM, OUT_DIM)),
    ("net.4.bias", (OUT_DIM,)),
]
TOTAL = sum(int(np.prod(shape)) for _, shape in LAYOUT)  # 4803


def bin_to_state_dict(bin_path):
    flat = np.fromfile(bin_path, dtype="<f4")
    if flat.size != TOTAL:
        raise SystemExit(f"{bin_path}: expected {TOTAL} float32 values "
                         f"({TOTAL * 4} bytes), found {flat.size}")
    state, i = {}, 0
    for name, shape in LAYOUT:
        n = int(np.prod(shape))
        block = flat[i:i + n].reshape(shape)
        i += n
        if name.endswith("weight"):
            block = block.T  # .bin stores (in, out); PyTorch wants (out, in)
        state[name] = torch.from_numpy(np.ascontiguousarray(block).copy())
    return state


def state_dict_to_flat(state):
    parts = []
    for name, _ in LAYOUT:
        a = state[name].detach().numpy()
        parts.append((a.T if name.endswith("weight") else a).flatten())
    return np.concatenate(parts).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="plugin weights file (.bin)")
    ap.add_argument("--out", required=True, help="PyTorch file to write (.pt)")
    args = ap.parse_args()

    state = bin_to_state_dict(args.bin)
    torch.save(state, args.out)

    original = np.fromfile(args.bin, dtype="<f4")
    round_trip = state_dict_to_flat(torch.load(args.out, map_location="cpu"))
    if not np.array_equal(original, round_trip):
        raise SystemExit("round-trip check FAILED: the .pt does not reproduce the .bin")
    print(f"Wrote {args.out} (round-trip check passed: identical to {args.bin})")


if __name__ == "__main__":
    main()
