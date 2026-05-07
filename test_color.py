import rasterio
import numpy as np

src = rasterio.open("resultados/super_resolved_2_5m_10bandas.tif")
arr = src.read()
crs = src.crs
transform = src.transform
src.close()

print(f"Shape: {arr.shape}")
for i in range(10):
    valid = arr[i][~np.isnan(arr[i])]
    if len(valid) > 0:
        print(f"  Band {i}: valid={len(valid)}/{arr[i].size}, range=[{valid.min():.4f}, {valid.max():.4f}]")
    else:
        print(f"  Band {i}: ALL NaN")

# Test RGB
rgb = arr[[2,1,0]].copy()  # B04, B03, B02
mask = np.isnan(rgb[0])
print(f"\nRGB valid: {np.sum(~mask)}/{rgb[0].size}")
rgb_clean = np.nan_to_num(rgb, nan=0.0)
print(f"RGB range: [{rgb_clean.min():.4f}, {rgb_clean.max():.4f}]")

# Test rio-color
from rio_color.operations import simple_atmo, saturation
vmin, vmax = rgb_clean.min(), rgb_clean.max()
if vmax > vmin:
    rgb_norm = (rgb_clean - vmin) / (vmax - vmin)
else:
    rgb_norm = np.clip(rgb_clean, 0, 1)

print(f"Norm range: [{rgb_norm.min():.4f}, {rgb_norm.max():.4f}]")

enh = simple_atmo(rgb_norm, haze=0.03, contrast=3, bias=0.5)
print(f"After atmo: [{enh.min():.4f}, {enh.max():.4f}]")
enh = saturation(enh, proportion=1.3)
enh = np.clip(enh, 0.0, 1.0)
print(f"After sat: [{enh.min():.4f}, {enh.max():.4f}]")

# Restore NaN
for b in range(3):
    enh[b][mask] = np.nan

# Save
p = "resultados/super_resolved_2_5m_cor.tif"
with rasterio.open(p, 'w', driver='GTiff',
                   height=enh.shape[1], width=enh.shape[2],
                   count=3, dtype=rasterio.float32,
                   crs=crs, transform=transform, compress='lzw') as dst:
    dst.write(enh.astype(np.float32))
print(f"Saved: {p}")
print("OK")