# -*- coding: utf-8 -*-
"""
Download de imagens Sentinel-2 L2A (todas as bandas) recortadas por polígono GPKG.
    - Lê o vetor, baixa o cubo que cobre toda a extensão (10m),
    - Recorta no polígono e salva como GeoTIFF.
"""

import os
import sys
import time
import socket
import numpy as np
import cubo
import rasterio
import rioxarray          # ATIVA o accessor .rio (obrigatório!)
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from shapely.ops import unary_union, transform as shapely_transform
import pyproj
import geopandas as gpd
from math import cos, radians

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
VETOR_LIMITE = "vetores/limite.gpkg"   # polígono da área de interesse
START_DATE   = "2024-09-08"
END_DATE     = "2025-09-08"
IMAGE_INDEX  = 0                       # qual imagem baixar (0 = primeira)
COLECAO      = "sentinel-2-l2a"
# Todas as bandas 10m, 20m e 60m – serão reamostradas para 10m
BANDAS       = ["B01","B02","B03","B04","B05","B06","B07","B08",
                "B8A","B09","B11","B12"]
RESOLUCAO_M  = 10
OUTPUT_DIR   = "resultados"
BLOB_HOST    = "sentinel2l2a01.blob.core.windows.net"

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

def compute_with_retry(da, max_tries=5, delay=2.0, backoff=2.0):
    """Baixa o cubo com tentativas em caso de falha de rede."""
    for attempt in range(1, max_tries + 1):
        try:
            # Seleciona a imagem desejada e baixa para numpy
            return da[IMAGE_INDEX].compute().to_numpy()
        except Exception as e:
            err = str(e).lower()
            if "resolve" in err or "host" in err or "timeout" in err or "connection" in err:
                print(f"  [!] Rede (tentativa {attempt}/{max_tries}): {e}")
            else:
                print(f"  [!] Erro (tentativa {attempt}/{max_tries}): {e}")
            if attempt < max_tries:
                d = delay * (backoff ** (attempt - 1))
                print(f"  Aguardando {d:.0f}s...")
                time.sleep(d)
            else:
                raise

def utm_epsg_from_lonlat(lon, lat):
    """Código EPSG da zona UTM com base na longitude e latitude."""
    zone = int((lon + 180) // 6) + 1
    return (32600 + zone) if lat >= 0 else (32700 + zone)

# =============================================================================
# PRINCIPAL
# =============================================================================
def main():
    print("=" * 60)
    print("DOWNLOAD SENTINEL-2 (todas as bandas) – 10m")
    print("=" * 60)

    # 1. Conectividade
    print("Verificando conectividade...")
    if not check_dns(BLOB_HOST) and not check_internet():
        print("  [!!] SEM INTERNET. Abortando.")
        return
    print("  [OK]\n")

    # 2. Vetor
    print(f"Lendo vetor: {VETOR_LIMITE}")
    gdf = gpd.read_file(VETOR_LIMITE)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    bbox = gdf.total_bounds  # [minx, miny, maxx, maxy]
    geometria = unary_union(gdf.geometry.values)
    print(f"  Bbox WGS84: {bbox}")

    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    print(f"  Centro: lat={cy:.6f}, lon={cx:.6f}")

    # 3. Tamanho do cubo
    lat_rad = radians(cy)
    m_per_deg_lon = 111320.0 * cos(lat_rad)
    m_per_deg_lat = 111320.0

    largura_deg = bbox[2] - bbox[0]
    altura_deg  = bbox[3] - bbox[1]
    largura_m = largura_deg * m_per_deg_lon
    altura_m  = altura_deg  * m_per_deg_lat

    edge_px = int(np.ceil(max(largura_m, altura_m) / RESOLUCAO_M))
    edge_px = int(edge_px * 1.1)  # 10% de folga
    print(f"  Área aprox: {largura_m:.0f}m × {altura_m:.0f}m")
    print(f"  Cubo: {edge_px}×{edge_px} pixels\n")

    # 4. Criação do cubo (todas as bandas, reamostragem para 10m)
    print("Criando cubo Sentinel-2 (todas as bandas)...")
    da = cubo.create(
        lat=cy,
        lon=cx,
        collection=COLECAO,
        bands=BANDAS,
        start_date=START_DATE,
        end_date=END_DATE,
        edge_size=edge_px,
        resolution=RESOLUCAO_M
    )

    # CRS e transform
    crs = da.rio.crs  # agora funciona pois rioxarray foi importado
    if crs is None:
        epsg = utm_epsg_from_lonlat(cx, cy)
        crs = CRS.from_epsg(epsg)
        print(f"  CRS inferido: EPSG:{epsg}")
    else:
        print(f"  CRS do cubo: {crs}")

    transform_cubo = da.rio.transform()
    print(f"  Transform: {transform_cubo}")

    # 5. Download da imagem selecionada (IMAGE_INDEX)
    cubo_np = compute_with_retry(da)
    print(f"  Shape do cubo baixado: {cubo_np.shape}")

    # 6. Reprojetar polígono para o CRS do cubo
    print("Reprojetando polígono para o CRS do cubo...")
    project = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    geometria_proj = shapely_transform(project, geometria)

    # 7. Máscara do polígono
    print("Aplicando máscara do polígono...")
    mask = geometry_mask(
        [geometria_proj],
        transform=transform_cubo,
        invert=True,
        out_shape=(cubo_np.shape[1], cubo_np.shape[2])
    )
    cubo_clip = cubo_np.copy().astype("float32")
    for b in range(cubo_clip.shape[0]):
        cubo_clip[b][~mask] = np.nan

    # 8. Salvar GeoTIFF
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "sentinel2_allbands_clipped.tif")

    print(f"Salvando GeoTIFF: {out_path}")
    with rasterio.open(
        out_path, 'w', driver='GTiff',
        height=cubo_clip.shape[1],
        width=cubo_clip.shape[2],
        count=cubo_clip.shape[0],
        dtype=rasterio.float32,
        crs=crs,
        transform=transform_cubo,
        compress='lzw'
    ) as dst:
        dst.write(cubo_clip.astype(rasterio.float32))

    print("\nDownload concluído com sucesso!")

if __name__ == "__main__":
    main()