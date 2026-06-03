"""
Standalone SEN2R super-resolution script.

Edit the CONSTANTS section below and run with:
  python sen2r.py

This script reads 4-band Sentinel-2 rasters (RGB+NIR) in a folder,
applies the SEN2SR model tile-by-tile, and saves super-resolved output files.
"""

import os
import math
import glob
import torch
import numpy as np
import rasterio
import mlstac

# =============================================================================
# CONSTANTES DE CONFIGURAÇÃO
# =============================================================================
INPUT_DIR =  r"D:\QgisProjects\teste_melhoirar_raster\ourives\20260530_23MMP_S2A"
PATTERN = "*.tif"
OUTPUT_DIR = "output_sen2r"
MODEL_DIR = "model/SEN2SRLite_RGBN"
USE_MAMBA = False
FATOR_SR = 4
TILE_SIZE = 128
OVERWRITE = False
COMPRESS = "lzw"

MODEL_URL_MAMBA = (
    "https://huggingface.co/tacofoundation/sen2sr/resolve/main"
    "/SEN2SR/NonReference_RGBN_x4/mlm.json"
)
MODEL_URL_LITE = (
    "https://huggingface.co/tacofoundation/sen2sr/resolve/main"
    "/SEN2SRLite/NonReference_RGBN_x4/mlm.json"
)

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def pad_to_multiple(arr, tile_size):
    _, H, W = arr.shape
    pad_h = (tile_size - H % tile_size) % tile_size
    pad_w = (tile_size - W % tile_size) % tile_size
    if pad_h > 0 or pad_w > 0:
        arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
    return arr, pad_h, pad_w


def save_geotiff(arr, transform, crs, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta = {
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "count": arr.shape[0],
        "dtype": rasterio.float32,
        "crs": crs,
        "transform": transform,
        "compress": COMPRESS,
    }
    with rasterio.open(path, 'w', **meta) as dst:
        dst.write(arr.astype(rasterio.float32))


def load_model(device):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_url = MODEL_URL_MAMBA if USE_MAMBA else MODEL_URL_LITE
    mlstac.download(file=model_url, output_dir=MODEL_DIR)
    model = mlstac.load(MODEL_DIR).compiled_model(device=device)
    return model


def enhance_tile(tile, model, device):
    tile = np.nan_to_num(tile, nan=0.0, posinf=0.0, neginf=0.0)
    tile = tile.astype(np.float32)
    if tile.max() > 2.0:
        tile = tile / 10000.0

    X = torch.from_numpy(tile).float().to(device)
    with torch.no_grad():
        out = model(X[None]).squeeze(0).cpu().numpy().astype(np.float32)
    return out


def process_file(path, model, device):
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}_sr.tif")
    if os.path.exists(out_path) and not OVERWRITE:
        print(f"Skipping existing: {out_path}")
        return

    with rasterio.open(path) as src:
        if src.count < 4:
            raise ValueError(f"Input raster must have at least 4 bands: {path}")

        arr = src.read([1, 2, 3, 4]).astype(np.float32)
        transform = src.transform
        crs = src.crs

        arr, pad_h, pad_w = pad_to_multiple(arr, TILE_SIZE)
        C, H, W = arr.shape
        nrows = H // TILE_SIZE
        ncols = W // TILE_SIZE

        out_arr = np.zeros((C, H * FATOR_SR, W * FATOR_SR), dtype=np.float32)

        print(f"Processing {path}: {nrows}x{ncols} tiles")
        for r in range(nrows):
            for c in range(ncols):
                h0, h1 = r * TILE_SIZE, (r + 1) * TILE_SIZE
                w0, w1 = c * TILE_SIZE, (c + 1) * TILE_SIZE
                tile = arr[:, h0:h1, w0:w1]
                sr_tile = enhance_tile(tile, model, device)
                h0s, h1s = r * TILE_SIZE * FATOR_SR, (r + 1) * TILE_SIZE * FATOR_SR
                w0s, w1s = c * TILE_SIZE * FATOR_SR, (c + 1) * TILE_SIZE * FATOR_SR
                out_arr[:, h0s:h1s, w0s:w1s] = sr_tile
                print(f"  tile {r * ncols + c + 1}/{nrows * ncols} OK")

        if pad_h or pad_w:
            out_arr = out_arr[:, : (H - pad_h) * FATOR_SR, : (W - pad_w) * FATOR_SR]

        out_transform = rasterio.Affine(
            transform.a / FATOR_SR,
            transform.b,
            transform.c,
            transform.d,
            transform.e / FATOR_SR,
            transform.f,
        )

        save_geotiff(out_arr, out_transform, crs, out_path)
        print(f"Saved super-resolved raster: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("SEN2R standalone super-resolution")
    print(f"Device: {device}")
    print(f"Input folder: {INPUT_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Pattern: {PATTERN}")
    print(f"Model dir: {MODEL_DIR}")

    model = load_model(device)
    files = sorted(glob.glob(os.path.join(INPUT_DIR, PATTERN)))
    if not files:
        print("No files found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for path in files:
        try:
            process_file(path, model, device)
        except Exception as exc:
            print(f"Error processing {path}: {exc}")


if __name__ == "__main__":
    main()
