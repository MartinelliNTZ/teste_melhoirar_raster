# -*- coding: utf-8 -*-
"""
SEN2SR - Super-resolução de imagens Sentinel-2 para 2.5m
Usa um arquivo vetorial (GPKG) como limite da Área de Interesse (AOI).

Saídas:
  - resultados/super_resolved_2_5m.tif      → 4 bandas (RGBN) super-resolvidas
  - resultados/super_resolved_2_5m_cor.tif   → RGB corrigido com rio-color
  - resultados/bandas_2_5m/*.tif             → TODAS as 10 bandas individuais
"""
import os
import sys
import time
import socket
import torch
import numpy as np
import mlstac
import cubo
import matplotlib.pyplot as plt
import rasterio
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from shapely.ops import unary_union
from scipy.ndimage import zoom as scipy_zoom

# =============================================================================
# CONFIGURAÇÕES (centralizadas no início)
# =============================================================================

VETOR_LIMITE = "vetores/limite.gpkg"
START_DATE   = "2026-04-08"
END_DATE     = "2026-05-07"
IMAGE_INDEX  = 0

COLECAO      = "sentinel-2-l2a"
RESOLUCAO_M  = 10
FATOR_SR     = 4        # 10m → 2.5m
TILE_SIZE    = 128
OUTPUT_DIR   = "resultados"
BLOB_HOST    = "sentinel2l2a01.blob.core.windows.net"

# Todas as 10 bandas do Sentinel-2 L2A
BANDAS = ["B02", "B03", "B04", "B08",   # 10m
          "B05", "B06", "B07", "B8A", "B11", "B12"]  # 20m

BANDAS_NOMES = [
    "B02_Blue", "B03_Green", "B04_Red",
    "B08_NIR", "B05_RedEdge1", "B06_RedEdge2",
    "B07_RedEdge3", "B8A_NarrowNIR", "B11_SWIR1", "B12_SWIR2"
]
BANDAS_10M_INDICES = [0, 1, 2, 3]   # B02, B03, B04, B08
BANDAS_20M_INDICES = [4, 5, 6, 7, 8, 9]  # B05..B12

# =============================================================================
# VERIFICAÇÕES DE DEPENDÊNCIAS
# =============================================================================

try:
    import rioxarray  # noqa: F401
except ImportError:
    sys.exit("pip install rioxarray")
try:
    import geopandas as gpd
except ImportError:
    sys.exit("pip install geopandas")
try:
    import mamba_ssm  # noqa: F401
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def check_dns(hostname, timeout=5):
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(hostname)
        return True
    except (socket.gaierror, OSError):
        return False
    finally:
        socket.setdefaulttimeout(None)

def check_internet():
    return any(check_dns(h) for h in [BLOB_HOST, "google.com", "huggingface.co"])

def compute_with_retry(da, idx=IMAGE_INDEX, max_tries=5, delay=2.0, backoff=2.0):
    for attempt in range(1, max_tries + 1):
        try:
            return da[idx].compute().to_numpy()
        except Exception as e:
            err = str(e).lower()
            if ("resolve" in err and "host" in err):
                print(f"\n  [!] DNS (tentativa {attempt}/{max_tries})")
                if attempt == 1 and not check_internet():
                    print("  [!!] SEM INTERNET.")
            elif "timeout" in err or "connection" in err:
                print(f"\n  [!] Rede (tentativa {attempt}/{max_tries})")
            else:
                print(f"\n  [!] Erro (tentativa {attempt}/{max_tries}): {e}")
            if attempt < max_tries:
                d = delay * (backoff ** (attempt - 1))
                print(f"  [~] Aguardando {d:.0f}s...")
                time.sleep(d)
            else:
                raise

def save_geotiff(arr, transform, crs, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, 'w', driver='GTiff',
                       height=arr.shape[1], width=arr.shape[2],
                       count=arr.shape[0], dtype=rasterio.float32,
                       crs=crs, transform=transform, compress='lzw') as dst:
        dst.write(arr.astype(rasterio.float32))

def pad_to_multiple(arr, tile_size):
    _, H, W = arr.shape
    pad_h = (tile_size - H % tile_size) % tile_size
    pad_w = (tile_size - W % tile_size) % tile_size
    if pad_h > 0 or pad_w > 0:
        arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
    return arr, pad_h, pad_w

def clip_to_polygon(arr, transform, geometria):
    mask = geometry_mask([geometria], transform=transform, invert=True,
                         out_shape=(arr.shape[1], arr.shape[2]))
    masked = arr.copy().astype("float32")
    for b in range(arr.shape[0]):
        masked[b][~mask] = np.nan
    return masked, mask

def upscale_bicubico(banda_2d, fator):
    """Upscale bicúbico de uma banda 2D por um fator."""
    return scipy_zoom(banda_2d, (fator, fator), order=3, mode='reflect')

def aplicar_rio_color(rgb_arr, transform, crs, caminho_saida):
    """Aplica rio-color (simple_atmo + saturation) e salva RGB."""
    from rio_color.operations import simple_atmo, saturation

    rgb = rgb_arr.copy().astype("float32")
    mask_nan = np.isnan(rgb[0])
    rgb = np.nan_to_num(rgb, nan=0.0)

    vmin, vmax = rgb.min(), rgb.max()
    if vmax > vmin:
        rgb_norm = (rgb - vmin) / (vmax - vmin)
    else:
        rgb_norm = rgb.copy()
    # Garantir valores estritamente dentro de (0,1) para o sigmoidal
    rgb_norm = np.clip(rgb_norm, 0.0001, 0.9999)

    rgb_enh = simple_atmo(rgb_norm, haze=0.03, contrast=3, bias=0.5)
    rgb_enh = saturation(rgb_enh, proportion=1.3)
    rgb_enh = np.clip(rgb_enh, 0.0, 1.0)

    for b in range(3):
        rgb_enh[b][mask_nan] = np.nan

    save_geotiff(rgb_enh, transform, crs, caminho_saida)
    return rgb_enh

def salvar_bandas_individuais(arr, nomes, transform, crs, pasta):
    """Salva cada banda como GeoTIFF individual."""
    pasta_bandas = os.path.join(OUTPUT_DIR, pasta)
    os.makedirs(pasta_bandas, exist_ok=True)
    for i, nome in enumerate(nomes):
        caminho = os.path.join(pasta_bandas, f"{nome}.tif")
        with rasterio.open(caminho, 'w', driver='GTiff',
                           height=arr.shape[1], width=arr.shape[2],
                           count=1, dtype=rasterio.float32,
                           crs=crs, transform=transform, compress='lzw') as dst:
            dst.write(arr[i:i+1].astype(rasterio.float32))
    print(f"  [{len(nomes)}] bandas salvas em: {pasta_bandas}/")

# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    print("=" * 60)
    print("SEN2SR - Super-resolução (10 bandas → 2.5m)")
    print("=" * 60)

    # Pré-verificação DNS
    print("\nVerificando conectividade...")
    if not check_dns(BLOB_HOST) and not check_internet():
        print("  [!!] SEM INTERNET. Abortando.")
        return
    print("  [OK]\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    # 1. Carregar vetor
    print(f"Lendo vetor: {VETOR_LIMITE}")
    gdf = gpd.read_file(VETOR_LIMITE)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    bbox = gdf.total_bounds
    geometria = unary_union(gdf.geometry.values)
    print(f"  Bbox: {bbox}\n")

    # 2. Calcular edge_size
    cy = (bbox[1] + bbox[3]) / 2.0
    cx = (bbox[0] + bbox[2]) / 2.0
    from math import cos, radians
    lat_rad = radians(cy)
    m_per_deg_lon = 111320.0 * cos(lat_rad)
    m_per_deg_lat = 111320.0

    largura_m = (bbox[2] - bbox[0]) * m_per_deg_lon
    altura_m  = (bbox[3] - bbox[1]) * m_per_deg_lat

    edge_px = int(max(np.ceil(largura_m / RESOLUCAO_M),
                      np.ceil(altura_m  / RESOLUCAO_M)))
    edge_px = ((edge_px + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
    edge_px = max(edge_px, TILE_SIZE)

    ncols = edge_px // TILE_SIZE
    nrows = edge_px // TILE_SIZE
    n_tiles_total = nrows * ncols
    print(f"Área: {largura_m:.0f}m × {altura_m:.0f}m")
    print(f"Cubo: {edge_px}×{edge_px} px ({ncols}×{nrows} tiles)")
    print(f"Bandas: {len(BANDAS)} ({', '.join(BANDAS)})\n")

    # 3. Seleção do modelo
    if HAS_MAMBA:
        model_url = ("https://huggingface.co/tacofoundation/sen2sr/resolve/main"
                     "/SEN2SR/NonReference_RGBN_x4/mlm.json")
        model_dir = "model/SEN2SR_RGBN"
        print("Arquitetura: Mamba (Full)")
    else:
        model_url = ("https://huggingface.co/tacofoundation/sen2sr/resolve/main"
                     "/SEN2SRLite/NonReference_RGBN_x4/mlm.json")
        model_dir = "model/SEN2SRLite_RGBN"
        print("Arquitetura: SwinIR (Lite)")

    os.makedirs(model_dir, exist_ok=True)
    mlstac.download(file=model_url, output_dir=model_dir)
    model = mlstac.load(model_dir).compiled_model(device=device)

    # 4. Baixar cubo
    print(f"\nBaixando cubo Sentinel-2 ({edge_px}×{edge_px}px)...")
    da = cubo.create(lat=cy, lon=cx, collection=COLECAO, bands=BANDAS,
                     start_date=START_DATE, end_date=END_DATE,
                     edge_size=edge_px, resolution=RESOLUCAO_M)

    crs = da.rio.crs
    transf_cubo = da.rio.transform()
    print(f"  CRS: {crs}")
    print(f"  Transform: {transf_cubo}")

    from rasterio.crs import CRS as RioCRS
    if crs is None:
        crs = RioCRS.from_epsg(32722)
        print(f"  [~] CRS forçado: {crs}")

    print("  [~] Reprojetando polígono...")
    from shapely.ops import transform as shapely_transform
    import pyproj
    project = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    geometria_proj = shapely_transform(project, geometria)

    cubo_np = compute_with_retry(da)
    print(f"  Shape: {cubo_np.shape}")

    cubo_np, pad_h, pad_w = pad_to_multiple(cubo_np, TILE_SIZE)
    if pad_h or pad_w:
        print(f"  Padding: {pad_h}×{pad_w}")

    # 5. Processar tiles com SEN2SR (apenas bandas 10m: B02,B03,B04,B08)
    print(f"\nProcessando {n_tiles_total} tiles com SEN2SR...")
    C, H, W = cubo_np.shape
    nrows = H // TILE_SIZE
    ncols = W // TILE_SIZE

    mosaico_sr_4b = np.zeros((4, nrows * TILE_SIZE * FATOR_SR,
                                 ncols * TILE_SIZE * FATOR_SR), dtype="float32")
    mosaico_orig  = np.zeros((C, nrows * TILE_SIZE, ncols * TILE_SIZE), dtype="float32")

    for r in range(nrows):
        for c in range(ncols):
            idx = r * ncols + c + 1
            h0, h1 = r * TILE_SIZE, (r + 1) * TILE_SIZE
            w0, w1 = c * TILE_SIZE, (c + 1) * TILE_SIZE
            print(f"  Tile {idx}/{n_tiles_total} [{r+1},{c+1}]...", end=" ")

            tile = cubo_np[:, h0:h1, w0:w1]
            tile_norm = (tile / 10000).astype("float32")

            # Reordenar para RGBN: [B04, B03, B02, B08]
            tile_rgbn = tile_norm[[2, 1, 0, 3], :, :]

            X = torch.from_numpy(tile_rgbn).float().to(device)
            X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            with torch.no_grad():
                sr = model(X[None]).squeeze(0)
            sr_np = sr.cpu().numpy().astype("float32")

            mosaico_orig[:, h0:h1, w0:w1] = tile_norm
            h0s, h1s = r * TILE_SIZE * FATOR_SR, (r + 1) * TILE_SIZE * FATOR_SR
            w0s, w1s = c * TILE_SIZE * FATOR_SR, (c + 1) * TILE_SIZE * FATOR_SR
            mosaico_sr_4b[:, h0s:h1s, w0s:w1s] = sr_np
            print("OK")

    print("  [OK] SEN2SR concluído!")

    # 6. Aplicar máscara
    print("\nAplicando máscara do polígono...")
    mosaic_orig_clipped, mask_orig = clip_to_polygon(mosaico_orig, transf_cubo, geometria_proj)

    transf_sr = rasterio.Affine(
        transf_cubo.a / FATOR_SR, transf_cubo.b, transf_cubo.c,
        transf_cubo.d, transf_cubo.e / FATOR_SR, transf_cubo.f,
    )

    mosaic_sr_4b_clipped, mask_sr = clip_to_polygon(mosaico_sr_4b, transf_sr, geometria_proj)

    # 7. Montar array com TODAS as 10 bandas em 2.5m
    print("\nMontando 10 bandas em 2.5m...")
    H_sr, W_sr = mosaic_sr_4b_clipped.shape[1], mosaic_sr_4b_clipped.shape[2]
    todas_bandas = np.zeros((10, H_sr, W_sr), dtype="float32")

    # Bandas super-resolvidas (mapear da ordem RGBN de volta para B02,B03,B04,B08)
    # mosaico_sr_4b_clipped: [0]=B04, [1]=B03, [2]=B02, [3]=B08
    todas_bandas[0] = mosaic_sr_4b_clipped[2]  # B02
    todas_bandas[1] = mosaic_sr_4b_clipped[1]  # B03
    todas_bandas[2] = mosaic_sr_4b_clipped[0]  # B04
    todas_bandas[3] = mosaic_sr_4b_clipped[3]  # B08
    print(f"  4 bandas 10m (SEN2SR): [B02,B03,B04,B08] OK")

    # Upscale bicúbico das bandas de 20m
    mosaic_orig_20m_clipped, _ = clip_to_polygon(mosaico_orig, transf_cubo, geometria_proj)
    for i, idx_orig in enumerate(BANDAS_20M_INDICES):
        idx_final = i + 4
        banda_up = upscale_bicubico(mosaic_orig_20m_clipped[idx_orig], FATOR_SR)
        banda_up[~mask_sr] = np.nan
        todas_bandas[idx_final] = banda_up.astype("float32")
        print(f"  20m → 2.5m: {BANDAS_NOMES[idx_orig]} OK")

    # 8. Salvar arquivos
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 8a. 4 bandas super-resolvidas (RGBN) — IGUAL ao geo3.py
    p_sr = os.path.join(OUTPUT_DIR, "super_resolved_2_5m.tif")
    save_geotiff(mosaic_sr_4b_clipped, transf_sr, crs, p_sr)
    print(f"\n[1] 4 bandas (RGBN): {p_sr}")

    # 8b. RGB corrigido com rio-color
    print("\n[2] Aplicando rio-color...")
    try:
        rgb_enh = aplicar_rio_color(
            mosaic_sr_4b_clipped[[0, 1, 2]], transf_sr, crs,
            os.path.join(OUTPUT_DIR, "super_resolved_2_5m_cor.tif")
        )
        print(f"    RGB corrigido salvo!")
        USAR_COLOR = True
    except Exception as e:
        print(f"    [!] rio-color falhou: {e}")
        USAR_COLOR = False

    # 8c. Bandas individuais (TODAS as 10)
    print(f"\n[3] Salvando bandas individuais...")
    salvar_bandas_individuais(todas_bandas, BANDAS_NOMES, transf_sr, crs, "bandas_2_5m")

    # 8d. Bandas individuais do RGB corrigido
    if USAR_COLOR:
        nomes_cor = ["B04_Red_cor", "B03_Green_cor", "B02_Blue_cor"]
        salvar_bandas_individuais(rgb_enh, nomes_cor, transf_sr, crs, "bandas_cor")

    # 9. Visualização
    print("\nGerando visualização...")
    fig, axes = plt.subplots(1, 3 if USAR_COLOR else 2, figsize=(16 if USAR_COLOR else 14, 7))

    axes[0].imshow(np.clip(mosaic_orig_clipped[[2, 1, 0]].transpose(1, 2, 0) * 1.5, 0, 1))
    axes[0].set_title("Original 10m")
    axes[0].axis('off')

    axes[1].imshow(np.clip(mosaic_sr_4b_clipped[[0, 1, 2]].transpose(1, 2, 0) * 1.5, 0, 1))
    axes[1].set_title("Super-res. 2.5m")
    axes[1].axis('off')

    if USAR_COLOR:
        axes[2].imshow(np.clip(rgb_enh.transpose(1, 2, 0), 0, 1))
        axes[2].set_title("Super-res. + rio-color")
        axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparacao.png"), dpi=150, bbox_inches='tight')
    plt.show(block=False)

    # Estatísticas
    area_km2 = np.sum(mask_orig) * RESOLUCAO_M**2 / 1e6
    print(f"\n--- Estatísticas ---")
    print(f"  Área poligonal: {area_km2:.2f} km²")
    print(f"  Shape original: {mosaic_orig_clipped.shape}")
    print(f"  Shape 4 bandas: {mosaic_sr_4b_clipped.shape}")
    print(f"  Shape 10 bandas: {todas_bandas.shape}")
    print(f"  10 bandas salvas em: {OUTPUT_DIR}/bandas_2_5m/")
    print("\n[OK] Concluído!")
    plt.show()


if __name__ == "__main__":
    main()