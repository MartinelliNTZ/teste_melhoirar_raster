# -*- coding: utf-8 -*-
"""
Lista imagens Sentinel‑2 com % de nuvens, permite escolher uma e baixar
todas as bandas (separadas) + RGB.
"""

import os
import socket
import time
import numpy as np
import geopandas as gpd
import cubo
import rasterio
import rioxarray
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from shapely.geometry import box
from shapely.ops import unary_union, transform as shapely_transform
import pyproj
from math import cos, radians
from pystac_client import Client

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
VETOR_LIMITE = "vetores/limite.gpkg"
START_DATE   = "2025-11-01"
END_DATE     = "2026-04-02"
CLOUD_LIMIT  = 0.5                     # filtrar imagens com nuvens ≤ 50%
BANDAS_ALL   = ["B01","B02","B03","B04","B05","B06","B07","B08",
                "B8A","B09","B11","B12"]
RGB_BANDS    = ["B04", "B03", "B02"]   # para montar o RGB
RESOLUTION   = 10
OUTPUT_DIR   = "result_8"
STAC_API     = "https://earth-search.aws.element84.com/v1"
COLLECTION   = "sentinel-2-l2a"

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================
def diagnose_vector(vetor_path):
    """Imprime diagnóstico detalhado do arquivo vetorial"""
    print(f"\n{'='*60}")
    print(f"DIAGNÓSTICO DO VETOR: {vetor_path}")
    print(f"{'='*60}")
    try:
        gdf = gpd.read_file(vetor_path)
        print(f"✓ Arquivo: {vetor_path}")
        print(f"  Número de features: {len(gdf)}")
        print(f"  CRS: {gdf.crs}")
        print(f"  Bounds (WGS84 ou nativo): {gdf.total_bounds}")
        print(f"  Tipos de geometria: {gdf.geometry.type.unique()}")
        print(f"  Áreas das feições:")
        for idx, row in gdf.iterrows():
            geom = row.geometry
            area = geom.area if hasattr(geom, 'area') else 'N/A'
            bounds = geom.bounds if hasattr(geom, 'bounds') else 'N/A'
            print(f"    [{idx}] tipo={geom.geom_type}, área={area}, bounds={bounds}")
        print(f"{'='*60}\n")
        return gdf
    except Exception as e:
        print(f"✗ ERRO ao ler vetor: {e}")
        print(f"{'='*60}\n")
        raise

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

def create_bbox_square(bbox, margin=0.05):
    """
    Cria um quadrado em graus que contém completamente o bounding box original.
    O quadrado é perfeitamente alinhado (sem rotação).
    
    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat]
        margin: margem adicional em fração do lado do quadrado (0.05 = 5%)
    
    Returns:
        bbox_quadrado: [min_lon, min_lat, max_lon, max_lat] (quadrado perfeito)
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Dimensões do bbox original
    width = max_lon - min_lon
    height = max_lat - min_lat
    
    # Lado do quadrado = maior dimensão + margem
    lado = max(width, height) * (1.0 + margin)
    
    # Centro do bbox original
    cx = (min_lon + max_lon) / 2.0
    cy = (min_lat + max_lat) / 2.0
    
    # Criar quadrado centrado
    lado_2 = lado / 2.0
    bbox_quadrado = [
        cx - lado_2,
        cy - lado_2,
        cx + lado_2,
        cy + lado_2
    ]
    
    print(f"\n  📐 Quadrado criado:")
    print(f"    Bbox original: {bbox}")
    print(f"    Lado do quadrado: {lado:.6f}°")
    print(f"    Bbox quadrado: {bbox_quadrado}")
    
    return bbox_quadrado

def save_square_vector(bbox_square, output_path="vetores/limite_quadrado.gpkg"):
    """Salva o quadrado intermediário em um arquivo vetorial se ainda não existir."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        print(f"  Vetor intermediário já existe: {output_path}")
        return output_path
    square_geom = box(*bbox_square)
    square_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[square_geom], crs="EPSG:4326")
    square_gdf.to_file(output_path, driver="GPKG")
    print(f"  Vetor intermediário salvo em: {output_path}")
    return output_path

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("CATÁLOGO SENTINEL-2 + DOWNLOAD (TODAS AS BANDAS + RGB)")
    print("=" * 60)

    # 1. Conectividade
    print("Verificando conectividade...")
    if not check_dns("earth-search.aws.element84.com") and not check_dns("google.com"):
        print("  [!!] SEM INTERNET. Abortando.")
        return
    print("  [OK]\n")

    # 2. Ler vetor
    print(f"Lendo vetor: {VETOR_LIMITE}")
    gdf = diagnose_vector(VETOR_LIMITE)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    bbox = gdf.total_bounds
    geometria = unary_union(gdf.geometry.values)
    print(f"  Bbox WGS84: {bbox}")

    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    print(f"  Centro: lat={cy:.6f}, lon={cx:.6f}")

    # 3. Criar quadrado que contém completamente o polígono
    bbox_square = create_bbox_square(bbox, margin=0.05)
    save_square_vector(bbox_square, output_path="vetores/limite_quadrado.gpkg")
    
    # Para o cubo.create(), vamos usar o bbox_square
    sq_min_lon, sq_min_lat, sq_max_lon, sq_max_lat = bbox_square
    sq_cx = (sq_min_lon + sq_max_lon) / 2.0
    sq_cy = (sq_min_lat + sq_max_lat) / 2.0
    sq_width = sq_max_lon - sq_min_lon
    sq_height = sq_max_lat - sq_min_lat
    
    # Tamanho em pixels (quadrado, portanto width == height)
    lat_rad = radians(sq_cy)
    m_per_deg_lon = 111320.0 * cos(lat_rad)
    m_per_deg_lat = 111320.0
    
    sq_width_m = sq_width * m_per_deg_lon
    sq_height_m = sq_height * m_per_deg_lat
    edge_px = int(np.ceil(max(sq_width_m, sq_height_m) / RESOLUTION))
    
    print(f"\n  Área do quadrado: {sq_width_m:.0f}m × {sq_height_m:.0f}m")
    print(f"  Cubo: {edge_px}×{edge_px} pixels")
    print(f"  Centro cubo: lat={sq_cy:.6f}, lon={sq_cx:.6f}\n")

    # 4. Buscar imagens no STAC
    print("🔎 Buscando imagens no catálogo STAC...")
    catalog = Client.open(STAC_API)
    search = catalog.search(
        collections=[COLLECTION],
        bbox=list(bbox),
        datetime=f"{START_DATE}/{END_DATE}",
        max_items=200
    )
    items = list(search.items())
    print(f"  {len(items)} cenas encontradas no total.\n")

    if not items:
        print("Nenhuma imagem disponível no período.")
        return

    # 5. Filtrar por nuvens
    valid_items = []
    print(f"{'Índice':<6} {'Data':<12} {'Nuvens %':<10} {'ID da cena'}")
    print("-" * 70)
    for item in items:
        cloud = item.properties.get("eo:cloud_cover", 100)
        if cloud <= CLOUD_LIMIT * 100:   # STAC retorna %
            valid_items.append(item)
            print(f"{len(valid_items):<6} {item.datetime.strftime('%Y-%m-%d'):<12} {cloud:5.1f}%     {item.id}")
    print("-" * 70)

    if not valid_items:
        print(f"Nenhuma imagem com nuvens ≤ {CLOUD_LIMIT*100:.0f}%.")
        return

    # 6. Seleção do usuário
    while True:
        try:
            idx = int(input(f"\nDigite o índice da imagem desejada (1 a {len(valid_items)}): "))
            if 1 <= idx <= len(valid_items):
                break
            print("  Índice fora do intervalo. Tente novamente.")
        except ValueError:
            print("  Digite um número inteiro.")

    selected_item = valid_items[idx - 1]
    print(f"\n✅ Imagem selecionada: {selected_item.id}")
    print(f"   Data: {selected_item.datetime.strftime('%Y-%m-%d')}")
    print(f"   Nuvens: {selected_item.properties.get('eo:cloud_cover'):.1f}%")

    # 7. Download usando cubo (passando collection, start_date, end_date)
    print("\n⏳ Baixando todas as bandas (10m)...")
    da = cubo.create(
        collection=COLLECTION,
        start_date=selected_item.datetime.strftime('%Y-%m-%d'),
        end_date=selected_item.datetime.strftime('%Y-%m-%d'),
        bands=BANDAS_ALL,
        resolution=RESOLUTION,
        lat=sq_cy,
        lon=sq_cx,
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

    # Remover dimensão temporal se existir
    if cubo_np.ndim == 4 and cubo_np.shape[0] == 1:
        cubo_np = cubo_np.squeeze(axis=0)
        print(f"  Squeeze (tempo): {cubo_np.shape}")
    
    # DIAGNÓSTICO: Verificar estrutura dos dados
    print(f"\n  [DIAGNÓSTICO] Dimensões do cubo:")
    print(f"    ndim: {cubo_np.ndim}")
    print(f"    shape: {cubo_np.shape}")
    
    if cubo_np.ndim == 4:
        print(f"    ⚠ AVISO: 4 dimensões (possível múltiplos sub-cubos/tiles)")
        print(f"      Dim 0: {cubo_np.shape[0]}")
        print(f"      Dim 1: {cubo_np.shape[1]}")
        print(f"      Dim 2: {cubo_np.shape[2]}")
        print(f"      Dim 3: {cubo_np.shape[3]}")
        print(f"\n    → Usando primeiro sub-cubo (índice 0)...")
        # Se temos múltiplos sub-cubos, usar o primeiro
        if cubo_np.shape[1] == len(BANDAS_ALL):
            # Estrutura: (n_cubos, bandas, altura, largura)
            cubo_np = cubo_np[0]
            print(f"      Extraído: {cubo_np.shape}")
        else:
            # Talvez seja (tempo, bandas, altura, largura)
            cubo_np = cubo_np.squeeze(axis=0)
            print(f"      Extraído (squeeze): {cubo_np.shape}")
    elif cubo_np.ndim == 3:
        print(f"    ✓ OK: 3 dimensões (bandas, altura, largura)")
        print(f"      Bandas: {cubo_np.shape[0]} (esperado: {len(BANDAS_ALL)})")
        print(f"      Altura: {cubo_np.shape[1]}")
        print(f"      Largura: {cubo_np.shape[2]}")
    print()

    # 8. Máscara do polígono
    print("Não aplicando máscara (mantendo dados do quadrado retangular)...")
    # Dados já estão prontos - mantém o quadrado completo sem recorte
    cubo_clip = cubo_np.copy().astype("float32")

    # 9. Salvar tudo
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Nome de arquivo baseado na data da cena
    date_tag = selected_item.datetime.strftime('%Y%m%d')
    collection_tag = 'st2'
    band_aliases = {
        'B01': 'COASTAL', 'B02': 'BLUE', 'B03': 'GREEN', 'B04': 'RED',
        'B05': 'RE', 'B06': 'RE', 'B07': 'RE', 'B08': 'NIR', 'B8A': 'NIR2',
        'B09': 'WATERVAPOR', 'B11': 'SWIR1', 'B12': 'SWIR2'
    }

    def band_filename(band):
        alias = band_aliases.get(band, '')
        alias_part = f'_{alias}' if alias else ''
        return f'{band}{alias_part}_{date_tag}_{collection_tag}.tif'

    # Bandas individuais
    print("\n💾 Salvando bandas individuais...")
    for i, banda in enumerate(BANDAS_ALL):
        fname = os.path.join(OUTPUT_DIR, band_filename(banda))
        with rasterio.open(
            fname, 'w', driver='GTiff',
            height=cubo_clip.shape[1], width=cubo_clip.shape[2],
            count=1, dtype=rasterio.float32,
            crs=crs, transform=transform_cubo, compress='lzw'
        ) as dst:
            dst.write(cubo_clip[i], 1)
        print(f"  {fname}")

    # RGB montado (B04,B03,B02)
    idx_rgb = [BANDAS_ALL.index(b) for b in RGB_BANDS]
    rgb = cubo_clip[idx_rgb]
    fname_rgb = os.path.join(OUTPUT_DIR, f'RGB_{date_tag}_{collection_tag}.tif')
    with rasterio.open(
        fname_rgb, 'w', driver='GTiff',
        height=rgb.shape[1], width=rgb.shape[2],
        count=3, dtype=rasterio.float32,
        crs=crs, transform=transform_cubo, compress='lzw'
    ) as dst:
        dst.write(rgb)
    print(f"  {fname_rgb}")

    # Stack final: se só vier RGB, não nomear como allbands
    if cubo_clip.shape[0] == len(BANDAS_ALL):
        fname_stack = os.path.join(OUTPUT_DIR, f'allbands_{date_tag}_{collection_tag}.tif')
        stack_count = len(BANDAS_ALL)
    elif cubo_clip.shape[0] == 3:
        fname_stack = os.path.join(OUTPUT_DIR, f'RGB_stack_{date_tag}_{collection_tag}.tif')
        stack_count = 3
    else:
        fname_stack = os.path.join(OUTPUT_DIR, f'stack_{date_tag}_{collection_tag}.tif')
        stack_count = cubo_clip.shape[0]

    with rasterio.open(
        fname_stack, 'w', driver='GTiff',
        height=cubo_clip.shape[1], width=cubo_clip.shape[2],
        count=stack_count, dtype=rasterio.float32,
        crs=crs, transform=transform_cubo, compress='lzw'
    ) as dst:
        dst.write(cubo_clip)
    print(f"  {fname_stack}")

    print("\n🎉 Download concluído! Arquivos salvos em:", os.path.abspath(OUTPUT_DIR))

if __name__ == "__main__":
    main()