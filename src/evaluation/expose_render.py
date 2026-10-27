"""
Post-process test_render.exr into a properly exposed, viewable PNG.

Switched from plain linear-exposure-then-clip to a Reinhard-style tonemap
(x / (1+x)) because this scene now has very high dynamic range: a small,
physically compact light produces a very bright direct highlight next to
a much dimmer diffuse-ish body. Plain linear exposure forces a choice
between a blown-out highlight or a near-black body -- it can't show both.
Reinhard tonemapping compresses the bright end smoothly instead of hard
clipping, so the dim body becomes visible without turning the highlight
into a flat white blob.
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np

# Repository edit (paths only): originally C:\NBrdfBuild\test_render.exr and
# C:\NBrdfBuild\test_render_exposed.png. Now relative to the current working
# directory, so run this from inside the render folder (where test_scene.rib
# writes test_render.exr).
IN_PATH = "test_render.exr"
OUT_PATH = "test_render_exposed.png"

# Pre-tonemap exposure multiplier -- still tune this, just less sensitive
# now since the tonemap compresses rather than clips. Try 60-200.
EXPOSURE = 120.0
GAMMA = 1.0 / 2.2

img = cv2.imread(IN_PATH, cv2.IMREAD_UNCHANGED)
if img is None:
    raise SystemExit(f"Could not load {IN_PATH}")

bgr = img[..., :3].astype(np.float32)

exposed = bgr * EXPOSURE
tonemapped_linear = exposed / (1.0 + exposed)          # Reinhard: compress highlights
tonemapped = np.clip(tonemapped_linear, 0.0, 1.0) ** GAMMA
png = (tonemapped * 255.0).astype(np.uint8)

cv2.imwrite(OUT_PATH, png)
print(f"Wrote {OUT_PATH} (exposure={EXPOSURE}, gamma={GAMMA:.3f}, tonemap=reinhard)")
print("If the body is still too dark, raise EXPOSURE. The highlight won't")
print("blow out to a flat dot the way it did before -- it'll just get a")
print("slightly bigger bright core as EXPOSURE increases.")
