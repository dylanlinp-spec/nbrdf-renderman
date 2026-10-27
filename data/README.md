# Data

This project trains on the **MERL BRDF database** (Matusik et al., 2003): reflectance measurements of 100 real materials. The data files are **not included** in this repository. They belong to Mitsubishi Electric Research Laboratories (MERL), and each one is 35 MB.

## Do I need it?

| Task | Needs MERL data? |
|---|---|
| Plugin build, renders, furnace tests, Table 3, timings | **No.** These use the trained weights in `results/weights/`. |
| Table 1, alum-bronze (as published) | No |
| Table 1, chrome (its ground-truth "GT" rows) | `chrome.binary` |
| Track A2 analytic reference (`render_gt_comparison.py`) | `alum-bronze.binary` **and** `chrome.binary` |
| Retraining the networks (`03_train_networks.sh`) | `alum-bronze.binary` and `chrome.binary` |

## How to get it

1. Go to MERL's BRDF page: https://www.merl.com/brdf. It links to the dataset's archive record at https://doi.org/10.5281/zenodo.8101680.
2. Read and accept the terms of use shown there.
3. Download the archive and unzip it.
4. Find these two files. In the archive the author used, they were in a folder named `brdfs`:
   - `alum-bronze.binary`
   - `chrome.binary`
5. Copy both into this folder, so you have:
   ```
   data/merl/alum-bronze.binary
   data/merl/chrome.binary
   ```
   (Create the `merl` folder if it does not exist.) To keep them somewhere else, set `MERL_DIR_DEFAULT` in `scripts/config.sh` instead.
6. Check them:
   ```bash
   bash scripts/01_check_setup.sh
   ```
   Each file must be exactly **34,992,012 bytes**. The loader also checks that the file starts with the grid size 90 × 90 × 180.

Treat these files as **read-only raw data**. Nothing in this project modifies them. Every script reads them and writes its results elsewhere.

`.gitignore` keeps `*.binary` files out of git, so they will never be uploaded by accident.

## Format, for the curious

Each `.binary` file starts with three 32-bit integers (90, 90, 180). These are the number of bins over the half-angle θh, the difference angle θd and the difference azimuth φd (Rusinkiewicz parameterisation). Then come 3 × 1,458,000 64-bit floats, all red values, then all green, then all blue. The loader scales them by (1/1500, 1.15/1500, 1.66/1500) and drops negative entries, which mark directions that were not measured. See `load_merl()` in `src/training/train_nbrdf.py`.
