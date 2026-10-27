# Known issues in the original results

These were found while assembling this repository, by checking the original build folder file by file. None of them is caused by the repository. The scripts reproduce the results **as published** and, where it matters, **also** produce a corrected version, so you can see both.

Issues 1 and 2 change what the paper can claim. The rest are minor.

---

## 1. The alum-bronze .pt and .bin are different networks

**What happened.** The alum-bronze network was trained, and its plugin file `nbrdf_weights.bin` was exported and used for every render. Later, the training notebook was run again. That run overwrote `nbrdf_weights_v2.pt` but did not re-export the `.bin`. PyTorch was never seeded ([issue 4](#4-training-is-not-seeded)), so the second run produced a different network.

**How it was checked.** Converting `nbrdf_weights_v2.pt` to plugin format does **not** reproduce `nbrdf_weights.bin`; the largest single weight differs by 5.59. For chrome, the same check reproduces the `.bin` exactly.

**What it affects.** These outputs used the `.pt`, so they describe a network that was **never rendered**:

| Output | Published value | Value for the rendered network |
|---|---|---|
| Table 1, alum-bronze max albedo | 0.1956 | **0.1886** |
| Track A2 analytic reference (`render_gt_comparison.py`) | MAE R/G/B 0.049 / 0.036 / 0.025 | run `07_run_analysis.sh`; see `a2_alum_compare_rendered_network.csv` |
| `furnace_alum_spherelight_compare.csv`, reference columns | about 0.130 at θ = 5.6° | about 0.128 at θ = 5.6° |

These outputs used the rendered network and are **not** affected: `furnace_alum_compare.csv`, Table 3 and 3b, all renders, and both timing results.

**Effect on the conclusions.** Small. Both networks conserve energy, and the furnace failure (up to 6× off) is far larger than the difference between them (about 4% in albedo). But Table 1 and the A2 numbers should be reported for the rendered network.

**What the scripts do.** `src/training/bin_to_pt.py` rebuilds a `.pt` directly from `nbrdf_weights.bin`. It checks that converting back gives the identical `.bin`. `07_run_analysis.sh` then:

- uses this rebuilt network as the reference for every furnace comparison, which matches how `furnace_alum_compare.csv` was originally made;
- writes `albedo_alum_rendered_network.csv` and `a2_alum_compare_rendered_network.csv`;
- still writes the as-published versions, so you can compare the two.

## 2. The plugin was built without compiler optimisation

**What happened.** The build command in the header of `src/plugin/NBrdf.cpp` has no `/O2` flag, so MSVC used its default, `/Od` (no optimisation). The original build folder supports this: the plugin's object file still contains separate copies of tiny internal helper functions (`Relu`, `Dot`, `Add`, `Normalize`). An optimised build would have inlined these and dropped them. This is strong evidence, not proof.

The benchmark `bench_eval.cpp` *was* built with `/O2`, as its header says.

**What it affects.**

- **Track C2 whole-render timings** (8.287 s, 5.20×, 3.89×) measure an *unoptimised* plugin. An optimised build will very likely be faster.
- **The C1 and C2 figures were measured on different builds.** C1 (1,292.96 ns per evaluation) used an optimised build, while the renderer in C2 used an unoptimised one.
- Any statement that the plugin was built with `cl /O2` or as a "release build" is inaccurate for the plugin.

**What the scripts do.** `04_build_plugin.sh` uses the original command, **without** `/O2`, so the published timings can be reproduced. To measure an optimised build as a separate, new experiment, add `/O2` to the `cl` line in that script, rebuild, and re-run `08_time_renders.sh`.

## 3. The θo = 0° row of the albedo tables varies slightly between computers

Re-running `compute_albedo.py` reproduces every row of `albedo_alum.csv` and `albedo_chrome.csv` **exactly, except θo = 0°**. That row differs by about 0.001 for alum-bronze and about 0.00004 for chrome, on a Linux machine with NumPy 2.4 and PyTorch 2.14. The cause is likely floating-point differences between maths libraries at this one angle. The maximum albedo (at 80°) is unaffected. The comparison tool reports this as **CLOSE**.

## 4. Training is not seeded

The training code seeds NumPy (`seed=0`, for the data split and batch sampling) but **not PyTorch**, so the network's starting weights are random on every run. Retraining therefore gives a network that is similar to the shipped one but not identical. This is why the shipped weights in `results/weights/` are the reference for every reproduced number.

`src/training/train_nbrdf.py` keeps the original behaviour by default. It adds an optional `--torch-seed` flag so that future runs can be made repeatable.

## 5. Only two materials have trained weights

The project notes mention three materials (alum-bronze, chrome and rubber). The original build folder holds trained weights for **alum-bronze and chrome only**, and every reported number uses one of those two. The training script works for any MERL material: pass a different `--material` file.

## 6. The furnace max |diff| depends on which light was used

Under the dome light (`furnace_alum_compare.csv`) the maximum difference is **0.2948**. Under the sphere light (`furnace_alum_spherelight_compare.csv`) it is **0.2923**. If a write-up quotes "0.292", that is the sphere-light figure. Either way the test fails by more than ten times the 0.02 bar.

## 7. The published `table3_variance.csv` has no `alpha` column

It was written by an earlier version of `sampler_variance.py`, before the α sweep was added. The current script always writes an `alpha` column. The values are identical: the reproduced file matches to 3.6 × 10⁻¹⁵.

## 8. The rendered Track B ablation shows no difference, by design of the bug

In the renderer, the three sampling modes give bit-identical images under the dome light, in both bin means and per-pixel noise. This is not a bug in the switch: the code path was checked end to end. It means `GenerateSample`'s contribution does not reach the image under `PxrDomeLight`, which matches the dome-light propagation problem recorded in the methodology (§3.4). That is why Track B was answered analytically, in Table 3, instead.

## 9. Open items, never run

- **Track A3 (convergence slope).** Never run, because the sampler is already known to be biased, not merely noisy.
- **Track C3 (network-width sweep).** Skipped for time.
- **A fix for the dome-light propagation defect.** Still open.

## 10. `NBrdf.args` is unverified

The parameter-description file was written to Pixar's documented format but never checked against a shipped example; its own comment says so. Command-line rendering with `.rib` files does not need it. It matters only for showing the plugin as a node in a DCC application such as Blender.
