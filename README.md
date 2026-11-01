# NBRDF for RenderMan

A small neural network that learned how to replicate measured material light reflectance behaviors, running inside Pixar's RenderMan as a native material plugin.

The network has 6 inputs, two hidden layers of 64 neurons and 3 outputs. It is trained on one material from the MERL measured-BRDF database. The file turns from a 35 MB table to a 19 KB file. A C++ Bxdf plugin then loads that file and evaluates the network millions of times per frame inside RenderMan 27.3.

This repository holds everything needed to reproduce the project's results including the training code, the plugin, performance testing, trained weights, and the original result tables. This repository assumes one has **never built a RenderMan plugin or trained a network before**.

> **Before you start, read [doc/known_issues.md](doc/known_issues.md).** It lists two things in the original results that need to be noted: the alum-bronze `.pt` and `.bin` files hold different networks, and the plugin was built without compiler optimization. Neither is hidden by this repository. The scripts reproduce the results as published and the corrected versions.

---

## Contents

1. [What you will reproduce](#1-what-you-will-reproduce)
2. [What you need](#2-what-you-need)
3. [Step-by-step instructions](#3-step-by-step-instructions)
4. [Checking your results](#4-checking-your-results)
5. [What is in this repository](#5-what-is-in-this-repository)
6. [How the pieces fit together](#6-how-the-pieces-fit-together)
7. [Citing, license, and credits](#7-citing-license-and-credits)

---

## 1. What you will reproduce

| Result | What it measures | Needs RenderMan? | Script | Published file |
|---|---|---|---|---|
| Table 1 | Energy conservation: directional albedo stays below 1 | No | `07_run_analysis.sh` | `results/tables/albedo_*.csv` |
| Test 1 | White-furnace render vs. the no-renderer albedo curve | Yes | `06` then `07` | `results/tables/furnace_*_compare.csv` |
| Test 2 | Distant-light render vs. a per-pixel analytic reference | Yes, plus MERL data | `06` then `07` | `results/tables/a2_alum_compare.csv` |
| Performance 1 | Sampling-mode ablation through the renderer | Yes | `06` then `07` | logs only |
| Table 3 / 3b | Noise of each sampling strategy, computed exactly | No | `07_run_analysis.sh` | `results/tables/table3_*.csv` |
| Performance 2 | Time per network evaluation, without RenderMan | No (needs MSVC) | `05_run_benchmark.sh` | numbers in this README |
| Performance 3 | Whole-render time vs. two stock RenderMan materials | Yes | `08_time_renders.sh` | `results/tables/timing.csv` |
| Figures | Analytic GT-vs-NBRDF spheres; two-sphere test render | Yes for the render | `07_run_analysis.sh` | `results/figures/` |

Specs: Intel Core Ultra 9 288V (8 cores, 3.3 GHz) with 32 GB RAM, Windows 11, RenderMan 27.3 and MSVC:

- **Table 1.** Maximum directional albedo is 0.1956 for alum-bronze (but see [known issue 1](doc/known_issues.md#1-the-alum-bronze-pt-and-bin-are-different-networks): the network actually rendered gives 0.1886). Chrome gives 0.1857. Both are well below 1.
- **Test 1.** The furnace test **fails**. Max |diff| is 0.295 for alum-bronze under the dome light and 0.292 under the sphere light, against a pass bar of 0.02. The sphere renders about 3× too bright near normal incidence and 3–6× too dark across the middle angles.
- **Performance 1.** MAE (R, G, B) is 0.049, 0.036 and 0.025.
- **Table 3** (α = 0.05). Cosine-only sampling has 6.9× to 838× the variance of the 50/50 mixture; GGX-only has 18× to 38×.
- **Performance 2** (median of 5 runs). `EvalReflectance` takes 1,292.96 ns. The network alone takes 1,211.24 ns and the angle conversion 77.38 ns.
- **Performance 3** (prman -t:8, median of 6 runs). NBRDF takes 8.287 s, PxrSurface 2.129 s and PxrDiffuse 1.594 s, so NBRDF is 5.20× the speed of PxrDiffuse and 3.89× PxrSurface.

---

## 2. What you need

### Computer

- **Windows 10 or 11, 64-bit.** The plugin has to be compiled with Microsoft's compiler, because RenderMan's own libraries are built with it. Training and the no-renderer analyses also work on macOS or Linux.
- About **3 GB of free disk space**: 1 GB for Python packages, 1 GB for RenderMan, and the rest for renders.
- Any modern CPU. A GPU is **not** needed.

### Software (all free)

| Software | Why | Where to get it |
|---|---|---|
| **Git for Windows** | Downloads this repository. It also gives you **Git Bash**, the terminal every script runs in. | https://git-scm.com/download/win |
| **Python 3.11** | Training and analysis. Versions 3.10 to 3.12 should also work. | https://www.python.org/downloads/ |
| **Build Tools for Visual Studio 2022** | The C++ compiler (`cl.exe`) for the plugin and benchmark | https://visualstudio.microsoft.com/visual-cpp-build-tools/ |
| **RenderMan 27.3** | The renderer. The free non-commercial licence is enough. | https://renderman.pixar.com/ |
| **MERL BRDF data** | Measured materials. Only needed for retraining and the ground-truth comparisons. | see [data/README.md](data/README.md) |

You do **not** need Blender, Maya, Houdini or a GPU.

---

## 3. Step-by-step instructions

Each step says how long it takes and what you should see. If something goes wrong, look up the exact error message in [doc/troubleshooting.md](doc/troubleshooting.md).

### Step 1. Install Git for Windows

1. Download the installer from https://git-scm.com/download/win and run it.
2. Click **Next** on every screen; no need to change the default. On the screen called **"Configuring the line ending conversions"**, the default ("Checkout Windows-style, commit Unix-style") is fine too, because this repository has a `.gitattributes` file that keeps the scripts working.
3. When it finishes, you will have a program called **Git Bash** in your Start menu.

* Should take about 5 minutes.*

### Step 2. Install Python

1. Download **Python 3.11** (Windows installer, 64-bit) from https://www.python.org/downloads/.
2. On the first installer screen, **tick "Add python.exe to PATH"** at the bottom. Then click **Install Now**.
3. Check that it worked. Open **Git Bash** and type:
   ```bash
   python --version
   ```
   You should see `Python 3.11.x`.
   - If you see `Python was not found; run without arguments to install from the Microsoft Store`, Windows is intercepting the command. Open **Settings → Apps → Advanced app settings → App execution aliases**, turn **off** both "python.exe" and "python3.exe", then close and reopen Git Bash.

*Should take about 5 minutes.*

### Step 3. Install the C++ compiler

1. Download **Build Tools for Visual Studio 2022** from https://visualstudio.microsoft.com/visual-cpp-build-tools/ and run it.
2. In the installer, tick **"Desktop development with C++"** and click **Install**. Leave the optional items at their defaults.
3. Restart your computer when it finishes.

*About 15–30 minutes of downloading.*

### Step 4. Install RenderMan

1. Create a free account at https://renderman.pixar.com/ and download **RenderMan Pro Server 27.3** for Windows, with the non-commercial licence. If 27.3 is no longer offered, take the closest 27.x and note the version you used.
2. Install it with the default options. The installer puts it somewhere similar to `C:\Program Files\Pixar\RenderManProServer-27.3`.
3. Follow Pixar's instructions to activate the licence.
4. Find out where it went. Open the Windows **Command Prompt** (not Git Bash) and type:
   ```
   echo %RMANTREE%
   ```
   - If this prints a folder path, RenderMan set itself up and you have nothing more to do here.
   - If it just prints `%RMANTREE%`, you will type the install folder into the settings file in Step 7.

*Takes about 15 minutes.*

### Step 5. Download this repository

1. Open **Git Bash**.
2. Go to where you keep projects. For example, to use your Documents folder or equivalent:
   ```bash
   cd ~/Documents
   ```
3. Download the repository. Replace `your_github_username` with the account that hosts it:
   ```bash
   git clone https://github.com/your_github_username/nbrdf-renderman.git
   cd nbrdf-renderman
   ```
4. **Every command from here on is typed in Git Bash, from inside this `nbrdf-renderman` folder.** If you close Git Bash, reopen it and `cd` back:
   ```bash
   cd ~/Documents/nbrdf-renderman
   ```
   Tip: in File Explorer you can right-click inside the folder and choose **Open Git Bash here** (on Windows 11, click **Show more options** first).

*About 1 minute.*

### Step 6. (Optional) Download the MERL data

You only need this to retrain the networks or to run the ground-truth comparisons (chrome Table 1 GT column and Test 2). Everything else uses the trained weights that ship with the repository.

Follow [data/README.md](data/README.md). You need `alum-bronze.binary` and `chrome.binary` in `data/merl/`.

*Takes about 5–10 minutes.*

### Step 7. Tell the scripts where things are

Open `scripts/config.sh` in any text editor (Notepad works). The rule is simple:

- `UPPER_CASE_WITH_UNDERSCORES` are setting names. Never change these.
- `lower_case_with_underscores` values are placeholders. Replace them with something from your computer.

The only line most people need to change:

```bash
RMANTREE_DEFAULT="path_to_your_renderman_install"
```

Change it to your RenderMan folder or equivalent from Step 4, **with forward slashes**:

```bash
RMANTREE_DEFAULT= <your_path_here>
```

If `echo %RMANTREE%` already printed a path in Step 4, you can leave the placeholder alone.

The other settings have working defaults:

| Setting | Default | Change it if... |
|---|---|---|
| `VCVARS64_BAT_DEFAULT` | found automatically | the compiler check in Step 8 fails |
| `WORK_DIR_DEFAULT` | `C:/NBrdfRepro` | you want builds and renders somewhere else. Use a path **without spaces**. |
| `MERL_DIR_DEFAULT` | `data/merl` in this repo | you keep the MERL files elsewhere |
| `PYTHON_BIN_DEFAULT` | `python` | `python --version` did not work but `py --version` does |

Save the file.

### Step 8. Check your setup

```bash
bash scripts/01_check_setup.sh
```

This changes nothing on your computer. It checks each requirement and prints `[ OK ]`, `[WARN]` or `[FAIL]`. Fix every `[FAIL]` line and run it again until it ends with **"All required checks passed"**. `[WARN]` lines are fine. For example, missing MERL data is a warning, because you only need it for some steps.

*Should take a few seconds.*

### Step 9. Set up Python packages

```bash
bash scripts/02_setup_python.sh
```

This creates a private Python environment in a folder called `.venv`, so nothing touches the rest of your computer. It then installs PyTorch, NumPy, Matplotlib, OpenCV and tifffile, and runs two self-tests. You should see lines ending in `[OK]` and `-> OK`.

*About 2–10 minutes; PyTorch is a large download.*

### Step 10. (Optional) Retrain the networks

```bash
bash scripts/03_train_networks.sh
```

This trains alum-bronze and chrome from the MERL data with the original code and settings: 20,000 steps, batch size 2,048, and a cosine learning-rate schedule. You will see the loss printed every 2,000 steps. The new weights go to `results/reproduced/retrained/`; the shipped weights are never overwritten.

The original training never fixed PyTorch's random seed, so **your networks will be similar but not bit-identical** to the shipped ones. All later steps use the shipped weights so that the published numbers can be reproduced exactly.

*Should take about 5–20 minutes per material on a laptop CPU.*

### Step 11. Build the plugin

```bash
bash scripts/04_build_plugin.sh
```

This compiles `src/plugin/NBrdf.cpp` into `NBrdf.dll`, in your work folder (`C:/NBrdfRepro` by default), using the exact compiler command in the source file's header. It should end with `Done. Built C:/NBrdfRepro/NBrdf.dll`.

*Takes under a minute.*

### Step 12. Run the speed benchmark (Performance 1)

```bash
bash scripts/05_run_benchmark.sh
```

This builds `bench_eval.exe`, then times 10 million network evaluations, 5 times. It prints the median of each timing line. Close other programs first, because anything else running slows it down.

*Should take about 1–2 minutes.*

### Step 13. Render the test scenes

```bash
bash scripts/06_render_scenes.sh
```

This renders 12 scenes, each written to an `.exr` image in the work folder. Each render prints RenderMan's progress, and the plugin prints a line along the lines of:

```
NBrdf: loaded 'C:/NBrdfRepro/nbrdf_weights.bin' (4803 floats, checksum ...)
```

If you see that line or something similar, the plugin found its weights. To render only some scenes, name them:

```bash
bash scripts/06_render_scenes.sh furnace_alum a2_distant_alum
```

*About 15–60 minutes in total. The 512-samples-per-pixel scenes are the slow ones.*

### Step 14. Run the analysis

```bash
bash scripts/07_run_analysis.sh
```

This produces every results table from the renders and the weights, and writes them to `results/reproduced/`. Steps whose inputs are missing (no MERL data, or a scene you did not render) are **skipped with a note**, not failed. It ends with a comparison against the published tables (see [Section 4](#4-checking-your-results)).

*About 5–10 minutes.*

### Step 15. Time the renders (Performance 3)

```bash
bash scripts/08_time_renders.sh
```

This renders the same scene with three materials, with 1 warm-up run then 6 timed runs each, and prints the ratios. Plug in your laptop and leave the computer alone while it runs.

*About 2–5 minutes.*

### Or: run everything at once

```bash
bash scripts/run_all.sh                 # steps 8, 9, 11-15
RETRAIN=yes bash scripts/run_all.sh     # also step 10
```

It stops at the first step that fails and tells you which script to re-run once you have fixed the problem.

---

## 4. Checking your results

`07_run_analysis.sh` finishes by printing a table like this. It is also saved to `results/reproduced/comparison_summary.txt`:

```
table                                    verdict      max |diff|
----------------------------------------------------------------
albedo_alum.csv                          CLOSE           0.00101
table3_alpha_sweep.csv                   MATCH          3.55e-15
table3_variance.csv                      MATCH          3.55e-15
...
```

| Verdict | Meaning |
|---|---|
| **MATCH** | Identical to 6 significant figures. |
| **CLOSE** | Max difference of 0.005 or less. Expected for anything rendered on a different computer, and for the θ = 0° row of the albedo tables ([known issue 3](doc/known_issues.md#3-the-θo--0-row-of-the-albedo-tables-varies-slightly-between-computers)). |
| **DIFFERENT** | Larger differences. See [doc/troubleshooting.md](doc/troubleshooting.md). One expected case: `furnace_alum_spherelight_compare.csv` ([known issue 1](doc/known_issues.md#1-the-alum-bronze-pt-and-bin-are-different-networks)). |
| **NOT RUN** | That step was skipped. |

The analysis also writes some files with no published counterpart. They are the **corrected** versions described in [doc/known_issues.md](doc/known_issues.md):

- `albedo_alum_rendered_network.csv` holds Table 1 for the alum-bronze network that was actually rendered.
- `a2_alum_compare_rendered_network.csv` holds Test 2 against a reference built from that same network.

Timing results depend on your hardware. Compare the **ratios** (NBRDF ÷ PxrDiffuse, and so on), not the seconds.

Every command's full output is saved in `results/reproduced/logs/`.

---

## 5. What is in this repository

```
nbrdf-renderman/
├── README.md                 you are here
├── LICENSE                   MIT licence for the code
├── CITATION.cff              how to cite this work
├── requirements.txt          Python packages
├── .gitignore / .gitattributes
├── data/
│   └── README.md             how to download the MERL files (they are not included)
├── doc/
│   ├── known_issues.md       READ THIS: what the original results get wrong or leave open
│   ├── troubleshooting.md    error messages and their fixes
│   ├── provenance.md         which original file each repo file came from, and every edit made
│   └── changelog.txt
├── scripts/                  run these, in number order (Git Bash)
│   ├── config.sh             the ONE file you edit
│   ├── lib.sh                helpers used by the scripts (do not run)
│   ├── 01_check_setup.sh
│   ├── 02_setup_python.sh
│   ├── 03_train_networks.sh  optional
│   ├── 04_build_plugin.sh
│   ├── 05_run_benchmark.sh
│   ├── 06_render_scenes.sh
│   ├── 07_run_analysis.sh
│   ├── 08_time_renders.sh
│   └── run_all.sh
├── scenes/                   RenderMan scene files (.rib), exactly as used
├── src/
│   ├── plugin/               NBrdf.cpp (the RenderMan plugin), NBrdf.args, compat/ headers
│   ├── benchmark/            bench_eval.cpp (timing without RenderMan)
│   ├── training/             train_nbrdf.py, bin_to_pt.py
│   └── evaluation/           every analysis script
└── results/
    ├── weights/              the exact trained weights behind every published number
    ├── tables/               the published result tables (never overwritten)
    ├── figures/              the published figures
    └── reproduced/           YOUR results appear here (ignored by git)
```

### The trained weights (`results/weights/`)

| File | What it is | Used by |
|---|---|---|
| `nbrdf_weights.bin` | alum-bronze, plugin format (19,212 bytes) | every alum-bronze render; Table 3; benchmark |
| `nbrdf_weights_chrome.bin` | chrome, plugin format | chrome renders |
| `nbrdf_weights_v2.pt` | alum-bronze, PyTorch format. **Not the same network as the .bin**; see [known issue 1](doc/known_issues.md#1-the-alum-bronze-pt-and-bin-are-different-networks). | published Table 1 (alum-bronze); published Test 2 reference |
| `nbrdf_weights_v2_chrome.pt` | chrome, PyTorch format. Identical network to its .bin. | Table 1 (chrome); Test 2 reference |
| `SHA256SUMS` | checksums; `01_check_setup.sh` verifies them | |

---

## 6. How the pieces fit together

```
  MERL .binary (35 MB)                               offline, Python
        │  src/training/train_nbrdf.py
        ▼
  6 → 64 → 64 → 3 network ──► .pt (PyTorch)  ──► compute_albedo.py, render_gt_comparison.py
        │  export (fixed order: W1 b1 W2 b2 W3 b3)
        ▼
  .bin (19,212 bytes) ──────────────────────────► sampler_variance.py, bench_eval.cpp
        │                                            runtime, C++
        ▼
  NBrdf.dll inside RenderMan (src/plugin/NBrdf.cpp)
        │  scenes/*.rib
        ▼
  .exr renders ──► furnace_check.py, a2_check.py, track_b_noise.py, time_render.py
```

The plugin answers the two questions a path tracer asks of every material:

- `GenerateSample` picks a new light direction itself. It uses a 50/50 mix of cosine-weighted and GGX (α = 0.05) sampling.
- `EvaluateSample` is given a light direction by the renderer.

Both report the same probability density, so that RenderMan's multiple importance sampling stays unbiased. A diagnostic parameter `samplingMode` (0 = mixture, the default; 1 = cosine only; 2 = GGX only) switches the sampler for the Performance 1 ablation without rebuilding.

---

## 7. Citing, license, and credits

If you use this work, please cite it using [CITATION.cff](CITATION.cff); GitHub shows a **"Cite this repository"** button for it. For long-term preservation, archive a release on Zenodo, which issues a DOI and integrates with GitHub.

- **Code** is released under the MIT licence ([LICENSE](LICENSE)).
- **MERL data** is not included. It belongs to Mitsubishi Electric Research Laboratories; see [data/README.md](data/README.md).
- **RenderMan** is a product of Pixar Animation Studios and is not included. The plugin uses its public `RixBxdf` API.

The method follows Sztrajman et al., *Neural BRDF Representation and Importance Sampling*, Computer Graphics Forum (2021). The training data is from Matusik et al., *A Data-Driven Reflectance Model*, ACM Transactions on Graphics (2003). GGX sampling follows Walter et al., *Microfacet Models for Refraction through Rough Surfaces*, EGSR (2007).

*Note: Track A1 is Test 1, Track A2 is Test 2, Track B is Performance 1, Track C1 is Performance 2, Track C3 is Performance 3, this is in case you get confused with file names vs this README. Additionally, generative AI was consulted to support the formatting, documentation, and the check the code of this respository.*
