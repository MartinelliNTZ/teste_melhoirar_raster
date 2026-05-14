# -*- coding: utf-8 -*-
"""
Lista imagens Sentinel‑2 com % de nuvens, permite escolher uma e baixar
todas as bandas (separadas) + RGB.
"""

import os
import socket
import numpy as np
import geopandas as gpd
import cubo
import rasterio
import rioxarray
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from shapely.ops import unary_union, transform as shapely_transform
import pyproj
from math import cos, radians
from pystac_client import Client

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
VETOR_LIMITE = "vetores/limite.gpkg"   # seu polígono
START_DATE   = "2024-09-08"
END_DATE     = "2025-09-08"
CLOUD_LIMIT  = 0.5                     # só mostra imagens com nuvens ≤ 50%
BANDAS_ALL   = ["B01","B02","B03","B04","B05","B06","B07","B08",
                "B8A","B09","B11","B12"]
RGB_BANDS    = ["B04", "B03", "B02"]   # para o RGB montado
RESOLUTION   = 10
OUTPUT_DIR   = "result_8"
STAC_API     = "https://earth-search.aws.element84.com/v1"
COLLECTION   = "sentinel-2-l2a"

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

def compute_with_retry(da, max_tries=5, delay=2.0, backoff=2.0):
    import time
    for attempt in range(1, max_tries + 1):
        try:
            return da.compute().to_numpy()
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ["resolve", "host", "timeout", "connection"]):
                print(f"  [!] Rede (tentativa {attempt}/{max_tries}): {e}")
            else:
                print(f"  [!] Erro (tentativa {attempt}/{max_tries}): {e}")
            if attempt < max_tries:
                d = delay * (backoff ** (attempt - 1))
                print(f"  Aguardando {d:.0f}s...")
                time.sleep(d)
            else:
                raise

def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("CATÁLOGO SENTINEL-2 + DOWNLOAD (TODAS AS BANDAS + RGB)")
    print("=" * 60)

    # 1. Verificar internet
    print("Verificando conectividade...")
    if not check_dns("earth-search.aws.element84.com") and not check_dns("google.com"):
        print("  [!!] SEM INTERNET. Abortando.")
        return
    print("  [OK]\n")

    # 2. Ler vetor
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

    # 3. Tamanho do cubo (pixels 10m)
    lat_rad = radians(cy)
    m_per_deg_lon = 111320.0 * cos(lat_rad)
    m_per_deg_lat = 111320.0
    largura_m = (bbox[2] - bbox[0]) * m_per_deg_lon
    altura_m  = (bbox[3] - bbox[1]) * m_per_deg_lat
    edge_px = int(np.ceil(max(largura_m, altura_m) / RESOLUTION))
    edge_px = int(edge_px * 1.1)  # 10% de folga
    print(f"  Área aprox: {largura_m:.0f}m × {altura_m:.0f}m")
    print(f"  Cubo: {edge_px}×{edge_px} pixels\n")

    # 4. Catálogo STAC (apenas para listar e filtrar nuvens)
    print("🔎 Buscando imagens no catálogo STAC...")
    catalog = Client.open(STAC_API)
    search = catalog.search(
        collections=[COLLECTION],
        bbox=list(bbox),
        datetime=f"{START_DATE}/{END_DATE}",
        max_items=200  # limite seguro
    )
    items = list(search.items())
    print(f"  {len(items)} cenas encontradas no total.\n")

    if not items:
        print("Nenhuma imagem disponível no período. Verifique datas ou área.")
        return

    # 5. Filtrar por nuvens ≤ CLOUD_LIMIT e montar tabela
    valid_items = []
    print(f"{'Índice':<6} {'Data':<12} {'Nuvens %':<10} {'ID da cena'}")
    print("-" * 70)
    for i, item in enumerate(items):
        cloud = item.properties.get("eo:cloud_cover", 100)  # padrão 100% se ausente
        if cloud <= CLOUD_LIMIT * 100:  # STAC retorna em % (0-100)
            valid_items.append(item)
            print(f"{len(valid_items):<6} {item.datetime.strftime('%Y-%m-%d'):<12} {cloud:5.1f}%     {item.id}")
    print("-" * 70)

    if not valid_items:
        print("Nenhuma imagem com nuvens ≤ 50% no período. Aumente o limite ou mude as datas.")
        return

    # 6. Seleção do usuário
    while True:
        try:
            idx = int(input(f"\nDigite o índice da imagem desejada (1 a {len(valid_items)}): "))
            if 1 <= idx <= len(valid_items):
                break
            print(f"  Índice fora do intervalo. Tente novamente.")
        except ValueError:
            print("  Digite um número inteiro.")

    selected_item = valid_items[idx - 1]
    print(f"\n✅ Imagem selecionada: {selected_item.id}")
    print(f"   Data: {selected_item.datetime.strftime('%Y-%m-%d')}")
    print(f"   Nuvens: {selected_item.properties.get('eo:cloud_cover'):.1f}%")

    # 7. Download usando cubo (item específico + recorte espacial fixo)
    print("\n⏳ Baixando todas as bandas (10m)...")
    da = cubo.create(
        item=selected_item,      # cena exata
        bands=BANDAS_ALL,
        resolution=RESOLUTION,
        lat=cy,
        lon=cx,
        edge_size=edge_px
    )

    # CRS e transform
    crs = da.rio.crs
    if crs is None:
        epsg = utm_epsg(cx, cy)
        crs = CRS.from_epsg(epsg)
    transform_cubo = da.rio.transform()

    cubo_np = compute_with_retry(da)
    print(f"  Shape bruto: {cubo_np.shape}")

    # 8. Máscara do polígono
    print("Aplicando máscara do polígono...")
    project = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    geometria_proj = shapely_transform(project, geometria)
    mask = geometry_mask(
        [geometria_proj],
        transform=transform_cubo,
        invert=True,
        out_shape=(cubo_np.shape[1], cubo_np.shape[2])
    )
    cubo_clip = cubo_np.copy().astype("float32")
    for b in range(cubo_clip.shape[0]):
        cubo_clip[b][~mask] = np.nan

    # 9. Salvar os resultados em result_8/
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 9a. Bandas individuais
    print("\n💾 Salvando bandas individuais...")
    for i, banda in enumerate(BANDAS_ALL):
        fname = os.path.join(OUTPUT_DIR, f"{banda}.tif")
        with rasterio.open(
            fname, 'w', driver='GTiff',
            height=cubo_clip.shape[1], width=cubo_clip.shape[2],
            count=1, dtype=rasterio.float32,
            crs=crs, transform=transform_cubo, compress='lzw'
        ) as dst:
            dst.write(cubo_clip[i], 1)
        print(f"  {fname}")

    # 9b. RGB montado (B04=índice 3, B03=índice 2, B02=índice 1)
    idx_rgb = [BANDAS_ALL.index(b) for b in RGB_BANDS]  # [3,2,1] (B04,B03,B02)
    rgb = cubo_clip[idx_rgb]
    fname_rgb = os.path.join(OUTPUT_DIR, "RGB.tif")
    with rasterio.open(
        fname_rgb, 'w', driver='GTiff',
        height=rgb.shape[1], width=rgb.shape[2],
        count=3, dtype=rasterio.float32,
        crs=crs, transform=transform_cubo, compress='lzw'
    ) as dst:
        dst.write(rgb)
    print(f"  {fname_rgb}")

    # 9c. Stack completo (todas as bandas juntas)
    fname_all = os.path.join(OUTPUT_DIR, "allbands.tif")
    with rasterio.open(
        fname_all, 'w', driver='GTiff',
        height=cubo_clip.shape[1], width=cubo_clip.shape[2],
        count=len(BANDAS_ALL), dtype=rasterio.float32,
        crs=crs, transform=transform_cubo, compress='lzw'
    ) as dst:
        dst.write(cubo_clip)
    print(f"  {fname_all}")

    print("\n🎉 Download concluído! Arquivos salvos em:", os.path.abspath(OUTPUT_DIR))

if __name__ == "__main__":
    main()