"""
Enhance rasters in a folder using rio_color (simple_atmo + saturation).

Usage:
  python enhance_rasters.py --input-dir path/to/folder [--pattern "*.tif"] [--proportion 1.3] [--overwrite]

This script reads each raster, extracts the first 3 bands (RGB), applies
simple_atmo + saturation from rio_color, then saves a new file with suffix
"_enhanced" (keeps additional bands if present).
"""

import os
import glob
import numpy as np
import rasterio
from rio_color.operations import simple_atmo, saturation

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
INPUT_DIR = r"D:\QgisProjects\teste_melhoirar_raster\ourives\20260530_23MMP_S2A"
PATTERN = "*.tif"
PROPORTION = 1.3
OVERWRITE = False
OUTPUT_SUFFIX = "_enhanced"

# =============================================================================


def enhance_rgb_array(rgb_arr, proportion=1.3):
    """Enhance an RGB array (3, H, W). Returns float32 in [0,1]."""
    if rgb_arr.shape[0] < 3:
        raise ValueError("rgb_arr must have 3 bands in axis 0")

    # Store mask of NaNs to restore later
    mask_nan = np.isnan(rgb_arr[0])

    # Replace NaN with 0 for processing
    arr = np.nan_to_num(rgb_arr, nan=0.0)

    # If values appear in reflectance scale (e.g. 0-10000), normalize
    vmin, vmax = arr.min(), arr.max()
    if vmax > 1.5 or vmin < 0.0:
        if vmax - vmin > 1e-6:
            norm = (arr - vmin) / (vmax - vmin)
        else:
            norm = np.clip(arr, 0.0, 1.0)
    else:
        norm = arr

    # Perform operations in float64 to match rio_color Cython buffer expectations
    norm64 = norm.astype(np.float64)
    try:
        enhanced = simple_atmo(norm64, haze=0.03, contrast=3, bias=0.5)
        enhanced = saturation(enhanced.astype(np.float64), proportion=proportion)
    except Exception:
        # As a fallback, try forcing float32 (some installs accept float32)
        enhanced = simple_atmo(norm64.astype(np.float32), haze=0.03, contrast=3, bias=0.5)
        enhanced = saturation(enhanced.astype(np.float32), proportion=proportion)

    enhanced = np.clip(enhanced, 0.0, 1.0)

    # Restore NaNs
    for b in range(3):
        enhanced[b][mask_nan] = np.nan

    return enhanced.astype(np.float32)


def process_file(path, out_path=None, proportion=1.3, overwrite=False):
    if out_path is None:
        base, ext = os.path.splitext(path)
        out_path = base + "_enhanced" + ext

    if os.path.exists(out_path) and not overwrite:
        print(f"Skipping existing: {out_path}")
        return out_path

    with rasterio.open(path) as src:
        meta = src.meta.copy()
        count = src.count

        if count < 3:
            print(f"Skipping (less than 3 bands): {path}")
            return None

        # Read first 3 bands as float
        rgb = src.read([1, 2, 3]).astype(np.float64)

        # If data likely in 0-10000 scale, normalize later inside enhancer
        enhanced_rgb = enhance_rgb_array(rgb, proportion=proportion)

        # Prepare output array: keep remaining bands unchanged (if any)
        if count > 3:
            other = src.read(list(range(4, count + 1))).astype(np.float32)
            out_arr = np.vstack([enhanced_rgb, other])
            meta.update({"count": out_arr.shape[0], "dtype": rasterio.float32})
        else:
            out_arr = enhanced_rgb
            meta.update({"count": 3, "dtype": rasterio.float32})

        # Ensure driver and compression
        meta.setdefault("driver", "GTiff")
        meta.update({"compress": "lzw"})

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(out_arr.astype(rasterio.float32))

    print(f"Saved: {out_path}")
    return out_path


def main():
    inp = INPUT_DIR
    pattern = PATTERN
    proportion = PROPORTION
    overwrite = OVERWRITE

    print(f"Using constants: INPUT_DIR={inp}, PATTERN={pattern}, PROPORTION={proportion}, OVERWRITE={overwrite}")
    files = sorted(glob.glob(os.path.join(inp, pattern)))
    if not files:
        print("No files found.")
        return

    for f in files:
        try:
            process_file(f, proportion=proportion, overwrite=overwrite)
        except Exception as e:
            print(f"Error processing {f}: {e}")


if __name__ == '__main__':
    main()
