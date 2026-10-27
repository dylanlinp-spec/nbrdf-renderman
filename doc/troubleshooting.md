# Troubleshooting

Find the message you saw (Ctrl+F) and follow the fix. Every command's full output is also saved in `results/reproduced/logs/`, which is the first place to look.

---

## Git Bash and the scripts

**`$'\r': command not found`** or **`syntax error near unexpected token $'\r'`**
: Git converted the scripts to Windows line endings. Fix it from inside the repository folder:
  ```bash
  git config core.autocrlf false
  git rm --cached -r . -q && git reset --hard
  ```
  The repository's `.gitattributes` prevents this on a fresh clone.

**`bash: scripts/01_check_setup.sh: No such file or directory`**
: You are not in the repository folder. Run `cd ~/Documents/nbrdf-renderman` (or wherever you cloned it) and try again. `ls` should list `README.md` and `scripts`.

**`ERROR: RMANTREE is not set`**
: Edit `scripts/config.sh` and replace `path_to_your_renderman_install` with your RenderMan folder, using forward slashes. See README step 7.

## Python

**`Python was not found; run without arguments to install from the Microsoft Store`**
: Open **Settings → Apps → Advanced app settings → App execution aliases** and turn off "python.exe" and "python3.exe". Close and reopen Git Bash.

**`python: command not found`**
: Python is not on your PATH. Reinstall it and tick **"Add python.exe to PATH"**. If `py --version` works, you can instead set `PYTHON_BIN_DEFAULT="py"` in `scripts/config.sh`.

**`The Python environment is missing. Run: bash scripts/02_setup_python.sh`**
: Do what it says. You only need to do this once.

**`ModuleNotFoundError: No module named 'torch'` (or `cv2`, `tifffile`)**
: You ran a script with the system Python instead of the project's. Use the numbered `bash scripts/...` commands, which pick the right Python. To run a Python file by hand, use `.venv/Scripts/python.exe path/to/script.py`.

**`could not load ... : cv2 failed and tifffile is not installed`**
: Some RenderMan installs write a TIFF file with an `.exr` name, and the scripts fall back to `tifffile` to read it. Re-run `bash scripts/02_setup_python.sh` to install it.

**`Got header (...) -- not a MERL .binary file?`** or **`AssertionError`** when loading MERL
: The `.binary` file is wrong or incomplete. Each one must be exactly 34,992,012 bytes; `01_check_setup.sh` checks this. Download it again (see `data/README.md`).

## Building (MSVC)

**`Could not find vcvars64.bat`**
: Build Tools for Visual Studio 2022 is not installed, or was installed without C++. Run the Visual Studio Installer, click **Modify**, tick **Desktop development with C++**, and install. If you installed it somewhere unusual, put the full path of `vcvars64.bat` in `VCVARS64_BAT_DEFAULT` in `scripts/config.sh`.

**`fatal error C1083: Cannot open include file: 'RixBxdf.h'`**
: `RMANTREE` points at the wrong folder. It must be the folder that contains `include\RixBxdf.h`, usually `C:/Program Files/Pixar/RenderManProServer-27.3`.

**`fatal error C1083: Cannot open include file: 'alloca.h'`**
: The `compat` folder was not copied. Re-run `bash scripts/04_build_plugin.sh`, which copies it every time.

**`error C3861: '__builtin_ctzll': identifier not found`**
: Same cause as the `alloca.h` error: `compat\gcc_builtins.h` must be force-included with `/FI`. Use the script; do not type the command by hand.

**`LINK : fatal error LNK1181: cannot open input file 'NBrdf.obj'`**
: Compilation failed earlier. Scroll up to the first `error` line. If you typed the command yourself: `/Fo` must be written directly against its file name, as `/FoNBrdf.obj` with no space.

**`LNK1104: cannot open file 'NBrdf.dll'`**
: The DLL is in use. Close any running `prman` or `it` (RenderMan's image viewer) and build again.

## Rendering (RenderMan)

**`prman was not found in .../bin`**
: `RMANTREE` is wrong; see the `RixBxdf.h` entry above.

**No `NBrdf: loaded ...` line, and the sphere is flat grey**
: The plugin ran but could not read its weights. In that case it falls back to a constant 0.5 so that the render still finishes. Look in the log for `NBrdf: failed to open weights file '...'`. The path it prints must exist. The scripts copy the weights into your work folder, so re-run `bash scripts/06_render_scenes.sh`.

**`NBrdf: weights file ... has wrong size` or `is LONGER than expected`**
: The `.bin` is not a 6→64→64→3 network (it must be 19,212 bytes). Re-download the repository, or re-export with `train_nbrdf.py`.

**An error saying the Bxdf "NBrdf" or plugin "NBrdf" was not found**
: RenderMan could not find `NBrdf.dll`. The scripts set `RMAN_RIXPLUGINPATH` to your work folder plus RenderMan's own plugin folder. Check that `NBrdf.dll` exists in the work folder; if not, run `bash scripts/04_build_plugin.sh`. If you run `prman` by hand, set the variable yourself first (in a Command Prompt):
  ```
  set RMAN_RIXPLUGINPATH=C:\NBrdfRepro;%RMANTREE%\lib\plugins
  ```

**A licence error from prman**
: Activate your RenderMan licence first, following Pixar's instructions. This is outside the scope of this repository.

**Renders are extremely slow**
: The 512-samples-per-pixel correctness scenes take minutes each on a laptop; this is expected. Render a few at a time by naming them, for example `bash scripts/06_render_scenes.sh furnace_alum`.

**`a2_check.py` warns that every sphere pixel in the render is exactly 0**
: The distant light points the wrong way for your RenderMan version. In `scenes/a2_distant_alum.rib` (and the two `b_distant_*` files), change `Rotate 41.00 0 1 0` to `Rotate -41.00 0 1 0`, as the scene's own comment explains. Note this in your write-up.

## Results

**A reproduced table says DIFFERENT**
: First check `doc/known_issues.md`. `furnace_alum_spherelight_compare.csv` is expected to differ (issue 1). Otherwise:
  - For **render-based** tables, check that you rendered with RenderMan 27.3 and the shipped weights. Different RenderMan versions can change sampling.
  - For **no-renderer** tables (albedo, Table 3), a large difference means the wrong weights file. Run `bash scripts/01_check_setup.sh`, which verifies the checksums.

**Benchmark or timing numbers are very different from the published ones**
: These depend on your CPU, power settings and background load, so compare the ratios. On a laptop, plug in, choose the "Best performance" power mode, and close other programs.
