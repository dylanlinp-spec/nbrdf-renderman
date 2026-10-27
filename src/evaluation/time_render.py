"""
Wall-clock render timer. Runs a render command several times and reports the
median and a robust spread (median absolute deviation), discarding a warm-up
run so first-run disk/cache effects don't skew the result.

Renderer-agnostic: it times the whole process with a monotonic clock, so it
works for `prman` or any other command. Use the SAME scene and sample count
for every material you compare (NBRDF vs PxrSurface vs PxrDiffuse); the only
thing that should differ between runs is the Bxdf.

Use {rib} as a placeholder for the scene file in the command template.

Examples:
    python time_render.py --rib nbrdf_scene.rib  --runs 6 --label NBRDF \
        -- prman -t:1 -Progress {rib}
    python time_render.py --rib pxrsurface_scene.rib --runs 6 --label PxrSurface \
        -- prman -t:1 {rib}

Pass --threads-note to record the thread flag you used; it's echoed into the
CSV row so the results table is self-documenting. Everything after `--` is the
command. Pure standard library.
"""
import argparse
import statistics
import subprocess
import sys
import time


def run_once(cmd):
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(f"command failed (exit {proc.returncode}): {' '.join(cmd)}")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rib", required=True, help="scene file; substituted for {rib} in the command")
    ap.add_argument("--runs", type=int, default=6, help="timed runs AFTER the warm-up (default 6)")
    ap.add_argument("--warmup", type=int, default=1, help="warm-up runs to discard (default 1)")
    ap.add_argument("--label", default="render", help="label for the output row (e.g. material/Bxdf)")
    ap.add_argument("--threads-note", default="", help="free-text note, e.g. 't:1' or '16 threads'")
    ap.add_argument("--csv", help="append a results row to this CSV")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="after '--', the render command; use {rib} for the scene file")
    args = ap.parse_args()

    cmd_tmpl = args.cmd
    if cmd_tmpl and cmd_tmpl[0] == "--":
        cmd_tmpl = cmd_tmpl[1:]
    if not cmd_tmpl:
        ap.error("provide the render command after '--'")
    cmd = [tok.replace("{rib}", args.rib) for tok in cmd_tmpl]

    print(f"command: {' '.join(cmd)}")
    for i in range(args.warmup):
        dt = run_once(cmd)
        print(f"  warm-up {i+1}: {dt:.3f}s (discarded)")

    times = []
    for i in range(args.runs):
        dt = run_once(cmd)
        times.append(dt)
        print(f"  run {i+1}: {dt:.3f}s")

    med = statistics.median(times)
    mad = statistics.median([abs(t - med) for t in times]) if len(times) > 1 else 0.0
    lo, hi = min(times), max(times)
    print(f"\n{args.label}: median {med:.3f}s  MAD {mad:.3f}s  (min {lo:.3f}, max {hi:.3f}, n={len(times)})")

    if args.csv:
        import csv
        import os
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as fh:
            wr = csv.writer(fh)
            if new:
                wr.writerow(["label", "median_s", "mad_s", "min_s", "max_s", "runs", "threads_note"])
            wr.writerow([args.label, f"{med:.3f}", f"{mad:.3f}",
                         f"{lo:.3f}", f"{hi:.3f}", len(times), args.threads_note])
        print(f"Appended row to {args.csv}")


if __name__ == "__main__":
    main()
