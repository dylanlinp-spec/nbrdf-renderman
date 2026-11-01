# Provenance: where every file came from

Every file below was copied from the original project build folder (`C:\NBrdfBuild`) or from the project's working copies of its scripts. The **Status** column compares each repository file's MD5 with the original's.

- **identical**: byte-for-byte the original.
- **paths only**: the only change is replacing hard-coded paths to the author's own folders (listed under "Edits" below). The computation is untouched.

The scene files (`scenes/*.rib`) are stored unchanged. When `06_render_scenes.sh` and `08_time_renders.sh` copy them into your work folder, they replace the author's folder name `C:/NBrdfBuild` with your `WORK_DIR`. That is the only change, and it happens only in the copy.

| Repository file | Original file | Status |
|---|---|---|
| `src/plugin/NBrdf.cpp` | `NBrdf_v3_diag.cpp` | identical |
| `src/plugin/NBrdf.args` | `NBrdf_1.args (named NBrdf.args in later notes)` | identical |
| `src/plugin/compat/alloca.h` | `compat/alloca.h` | identical |
| `src/plugin/compat/gcc_builtins.h` | `compat/gcc_builtins.h` | identical |
| `src/benchmark/bench_eval.cpp` | `bench_eval.cpp` | identical |
| `src/evaluation/compute_albedo.py` | `compute_albedo.py` | identical |
| `src/evaluation/furnace_check.py` | `furnace_check.py` | identical |
| `src/evaluation/a2_check.py` | `a2_check.py` | identical |
| `src/evaluation/track_b_noise.py` | `track_b_noise.py` | identical |
| `src/evaluation/sampler_variance.py` | `sampler_variance.py` | identical |
| `src/evaluation/time_render.py` | `time_render.py` | identical |
| `src/evaluation/render_gt_comparison.py` | `render_gt_comparison.py` | paths only |
| `src/evaluation/expose_render.py` | `expose_render.py` | paths only |
| `scenes/a2_distant_alum.rib` | `a2_distant_alum.rib` | identical |
| `scenes/b16_furnace_alum_cosine.rib` | `b16_furnace_alum_cosine.rib` | identical |
| `scenes/b16_furnace_alum_ggx.rib` | `b16_furnace_alum_ggx.rib` | identical |
| `scenes/b16_furnace_alum_mixture.rib` | `b16_furnace_alum_mixture.rib` | identical |
| `scenes/b_distant_alum_cosine.rib` | `b_distant_alum_cosine.rib` | identical |
| `scenes/b_distant_alum_ggx.rib` | `b_distant_alum_ggx.rib` | identical |
| `scenes/c2_nbrdf.rib` | `c2_nbrdf.rib` | identical |
| `scenes/c2_pxrdiffuse.rib` | `c2_pxrdiffuse.rib` | identical |
| `scenes/c2_pxrsurface.rib` | `c2_pxrsurface.rib` | identical |
| `scenes/furnace_alum.rib` | `furnace_alum.rib` | identical |
| `scenes/furnace_alum_cosine.rib` | `furnace_alum_cosine.rib` | identical |
| `scenes/furnace_alum_ggx.rib` | `furnace_alum_ggx.rib` | identical |
| `scenes/furnace_alum_spherelight.rib` | `furnace_alum_spherelight.rib` | identical |
| `scenes/furnace_chrome.rib` | `furnace_chrome.rib` | identical |
| `scenes/test_scene.rib` | `test_scene.rib` | identical |
| `results/weights/nbrdf_weights.bin` | `nbrdf_weights.bin` | identical |
| `results/weights/nbrdf_weights_chrome.bin` | `nbrdf_weights_chrome.bin` | identical |
| `results/weights/nbrdf_weights_v2.pt` | `nbrdf_weights_v2.pt` | identical |
| `results/weights/nbrdf_weights_v2_chrome.pt` | `nbrdf_weights_v2_chrome.pt` | identical |
| `results/tables/a2_alum_compare.csv` | `a2_alum_compare.csv` | identical |
| `results/tables/albedo_alum.csv` | `albedo_alum.csv` | identical |
| `results/tables/albedo_chrome.csv` | `albedo_chrome.csv` | identical |
| `results/tables/furnace_alum_compare.csv` | `furnace_alum_compare.csv` | identical |
| `results/tables/furnace_alum_spherelight_compare.csv` | `furnace_alum_spherelight_compare.csv` | identical |
| `results/tables/furnace_chrome_compare.csv` | `furnace_chrome_compare.csv` | identical |
| `results/tables/table3_alpha_sweep.csv` | `table3_alpha_sweep.csv` | identical |
| `results/tables/table3_variance.csv` | `table3_variance.csv` | identical |
| `results/tables/timing.csv` | `timing.csv` | identical |
| `results/figures/gt_vs_nbrdf.png` | `gt_vs_nbrdf.png` | identical |
| `results/figures/test_render_exposed.png` | `test_render_exposed.png` | identical |

## Edits

1. **`src/evaluation/render_gt_comparison.py`**: `MATERIALS` and `OUT_PATH` pointed to `C:\Users\dylan\Downloads\BRDFDatabase\...` and `C:\NBrdfBuild\...`. They now point to `data/merl/` and `results/weights/` inside the repository, can be overridden with the environment variables `MERL_DIR`, `WEIGHTS_DIR` and `ALUM_BRONZE_PT`, and write `gt_vs_nbrdf.png` to the current folder. The default `.pt` files are the same ones the original used.
2. **`src/evaluation/expose_render.py`**: `IN_PATH` and `OUT_PATH` were `C:\NBrdfBuild\test_render.exr` and `...test_render_exposed.png`. They are now the same file names, relative to the current folder.

## New files (not in the original project)

| File | Why it was added |
|---|---|
| `src/training/train_nbrdf.py` | A command-line version of `train_nbrdf_v2.ipynb` and `train_nbrdf_chrome.ipynb` (those two notebooks differ only in their paths). Checked with Python's syntax-tree comparison: `load_merl`, `angles_to_input`, `NBRDF`, `train` and `export_weights_for_renderman` are **identical** to the notebook. `plot_slice_comparison` differs only in saving its figure to a file instead of showing it. Adds an optional `--torch-seed`, off by default. |
| `src/training/bin_to_pt.py` | Converts a plugin `.bin` back to a `.pt`, with a round-trip check. Needed because of known issue 1. |
| `src/evaluation/compare_with_published.py` | Compares reproduced tables with the published ones. |
| `scripts/*.sh` | Driver scripts that run each step with the original commands and settings. |
| `results/weights/SHA256SUMS` | Checksums for the shipped weights. |

## Original files deliberately left out

These were part of the project's development but produced none of the reported results:

| Original file(s) | Why it is not here |
|---|---|
| `NBrdf_stub.cpp` to `NBrdf_stub7.cpp`, `test_scene_stub*.rib`, `bogus_scene.rib` | The seven-stage bisection used to find the original crash (a lobe written before `AddActiveLobe()`). Superseded. |
| `NBrdf.cpp` (v1, 21-wide), `NBrdf3.cpp`, `NBrdf_v2.cpp` | Earlier plugin versions. `NBrdf_v3_diag.cpp` with `samplingMode = 0` behaves exactly as v2, as its header documents. |
| `train_nbrdf.py`, `train_nbrdf.ipynb` (v1, 21-wide, 3,000 steps) | The first network, replaced by the 64-wide v2 used for every result. |
| `check_render.py`, `check_symmetry.py`, `find_highlight.py` | One-off visual diagnostics from debugging the test render. |
| `control_scene.rib` | A PxrDiffuse control used while isolating the crash. |
| `convergence.py` | Written for Test 3, which was never run (known issue 9). |
| `studio.hdr`, `studio.tex` | An HDRI used during early debugging only (48 MB). No reported scene uses it. |
| `nbrdf_weights.pt` | The v1 (21-wide) network. |
| `*.exr` renders, `*.npy` | Regenerated by the scripts. |

