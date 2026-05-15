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
from rasterio.transform import xy
from shapely.geometry import box, shape
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
CLOUD_LIMIT  = 0.5                   # filtrar imagens com nuvens ≤ 50%
STRICT_TILE_CONTAINMENT = True        # se True, mantém só cenas que cobrem todo o talhão
CROP_TO_TALHAO = False                # se True, recorta o raster final pelo talhão
DOWNLOAD_FULL_TILE = True            # se True, baixa quadrado inteiro sem recorte (ignora CROP_TO_TALHAO)
DOWNLOAD_FULL_STAC_TILE = False      # se True, baixa o footprint inteiro do STAC item (~110km x 110km), ignora polígono
BANDAS_ALL   = ["B01","B02","B03","B04","B05","B06","B07","B08",
                "B8A","B09","B11","B12"]
RGB_BANDS    = ["B04", "B03", "B02"]   # para montar o RGB
RESOLUTION   = 10                    # resolução em metros (reduz para 5 para maior detalhe/pixels)
OUTPUT_DIR   = "result_8"
STAC_API     = "https://earth-search.aws.element84.com/v1"
COLLECTION   = "sentinel-2-l2a"
BBOX_MARGIN  = 0.05                  # margem ao redor do talhão (0.05 = 5%, aumente para 0.2 = 20% para maior cobertura)

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
    Cria um quadrado que contém completamente o bounding box original.
    Funciona com coordenadas projetadas (metros) ou graus.
    
    Args:
        bbox: [min_x, min_y, max_x, max_y]
        margin: margem adicional em fração do lado do quadrado
    
    Returns:
        bbox_quadrado: [min_x, min_y, max_x, max_y] (quadrado perfeito)
    """
    min_x, min_y, max_x, max_y = bbox
    
    # Dimensões do bbox original
    width = max_x - min_x
    height = max_y - min_y
    
    # Lado do quadrado = maior dimensão + margem
    lado = max(width, height) * (1.0 + margin)
    
    # Centro do bbox original
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    
    # Criar quadrado centrado
    lado_2 = lado / 2.0
    bbox_quadrado = [
        cx - lado_2,
        cy - lado_2,
        cx + lado_2,
        cy + lado_2
    ]
    
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

def item_covers_polygon(item_geom, talhao_geom):
    """Retorna True se a geometria do item STAC cobre completamente o talhão."""
    if item_geom is None:
        return False
    try:
        item_shape = shape(item_geom)
        return item_shape.covers(talhao_geom)
    except Exception:
        return False

def filter_items_by_containment(items, talhao_geom):
    filtered = []
    for item in items:
        if item_covers_polygon(item.geometry, talhao_geom):
            filtered.append(item)
    return filtered

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
    print(f"  Extensão: X=[{bbox[0]:.6f}, {bbox[2]:.6f}] Y=[{bbox[1]:.6f}, {bbox[3]:.6f}]")

    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    print(f"  Centro: lat={cy:.6f}, lon={cx:.6f}")
    
    # 2.1 Converter para UTM para calculo preciso
    epsg_utm = utm_epsg(cx, cy)  # Detectar EPSG UTM correto
    print(f"  CRS UTM detectado: EPSG:{epsg_utm}")
    
    gdf_utm = gdf.to_crs(f"EPSG:{epsg_utm}")
    bbox_utm = gdf_utm.total_bounds
    geometria_utm = unary_union(gdf_utm.geometry.values)
    
    print(f"\n  Bbox UTM (EPSG:{epsg_utm}): {bbox_utm}")
    print(f"  Extensão UTM: X=[{bbox_utm[0]:.2f}, {bbox_utm[2]:.2f}] Y=[{bbox_utm[1]:.2f}, {bbox_utm[3]:.2f}]")
    
    orig_width_m_utm = bbox_utm[2] - bbox_utm[0]
    orig_height_m_utm = bbox_utm[3] - bbox_utm[1]
    print(f"  Tamanho do polígono (UTM): {orig_width_m_utm:.0f}m (E-W) × {orig_height_m_utm:.0f}m (N-S)")

    # 3. Criar quadrado em UTM (coordenadas precisas em metros)
    bbox_square_utm = create_bbox_square(bbox_utm, margin=BBOX_MARGIN)
    print(f"\n  📐 Quadrado criado (UTM):")
    print(f"    Lado do quadrado: {max(bbox_square_utm[2]-bbox_square_utm[0], bbox_square_utm[3]-bbox_square_utm[1]):.0f}m")
    print(f"    Bbox quadrado UTM: X=[{bbox_square_utm[0]:.2f}, {bbox_square_utm[2]:.2f}] Y=[{bbox_square_utm[1]:.2f}, {bbox_square_utm[3]:.2f}]")
    
    # Converter quadrado de volta para WGS84
    transformer_utm_to_wgs84 = pyproj.Transformer.from_crs(f"EPSG:{epsg_utm}", "EPSG:4326", always_xy=True)
    min_lon_sq, min_lat_sq = transformer_utm_to_wgs84.transform(bbox_square_utm[0], bbox_square_utm[1])
    max_lon_sq, max_lat_sq = transformer_utm_to_wgs84.transform(bbox_square_utm[2], bbox_square_utm[3])
    bbox_square = [min_lon_sq, min_lat_sq, max_lon_sq, max_lat_sq]
    
    save_square_vector(bbox_square, output_path="vetores/limite_quadrado.gpkg")
    
    # Para o cubo.create(), vamos usar o bbox_square em WGS84
    sq_min_lon, sq_min_lat, sq_max_lon, sq_max_lat = bbox_square
    print(f"\n  Bbox QUADRADO (WGS84): [minLon={sq_min_lon:.6f}, minLat={sq_min_lat:.6f}, maxLon={sq_max_lon:.6f}, maxLat={sq_max_lat:.6f}]")
    print(f"  Extensão quadrado WGS84: X=[{sq_min_lon:.6f}, {sq_max_lon:.6f}] Y=[{sq_min_lat:.6f}, {sq_max_lat:.6f}]")
    
    sq_cx = (sq_min_lon + sq_max_lon) / 2.0
    sq_cy = (sq_min_lat + sq_max_lat) / 2.0
    
    # Dimesões do quadrado em UTM (já calculadas)
    sq_width_m = bbox_square_utm[2] - bbox_square_utm[0]
    sq_height_m = bbox_square_utm[3] - bbox_square_utm[1]
    edge_px = int(np.ceil(max(sq_width_m, sq_height_m) / RESOLUTION))
    
    print(f"  Dimensões do quadrado: {sq_width_m:.0f}m (E-W) × {sq_height_m:.0f}m (N-S)")
    print(f"  Resolução: {RESOLUTION}m/pixel")
    print(f"  Cubo solicitado: {edge_px}×{edge_px} pixels = {edge_px*RESOLUTION:.0f}m × {edge_px*RESOLUTION:.0f}m")
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
    candidates = []
    for item in items:
        cloud = item.properties.get("eo:cloud_cover", 100)
        if cloud <= CLOUD_LIMIT * 100:   # STAC retorna %
            candidates.append(item)

    if STRICT_TILE_CONTAINMENT:
        retained = filter_items_by_containment(candidates, geometria)
        print(f"  {len(retained)} cenas com nuvens ≤ {CLOUD_LIMIT*100:.0f}% e talhão totalmente dentro do tile.")
        candidates = retained
        if not candidates:
            print("Nenhuma imagem atende ao critério de contenção total do polígono.")
            return

    valid_items = candidates
    print(f"{'Índice':<6} {'Data':<12} {'Nuvens %':<10} {'ID da cena'}")
    print("-" * 70)
    for idx, item in enumerate(valid_items, 1):
        cloud = item.properties.get("eo:cloud_cover", 100)
        print(f"{idx:<6} {item.datetime.strftime('%Y-%m-%d'):<12} {cloud:5.1f}%     {item.id}")
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

    # 6.1 Determinar área a baixar
    if DOWNLOAD_FULL_STAC_TILE:
        print("\n🌐 Modo FULL STAC TILE: usando footprint inteiro do item Sentinel-2...")
        # Extrair bounds do footprint do STAC item
        if selected_item.geometry:
            geom_stac = shape(selected_item.geometry)
            bounds_stac = geom_stac.bounds  # (minx, miny, maxx, maxy)
            print(f"  📍 Footprint STAC (WGS84):")
            print(f"    Geometry type: {geom_stac.geom_type}")
            print(f"    Bounds: minX={bounds_stac[0]:.6f}, minY={bounds_stac[1]:.6f}, maxX={bounds_stac[2]:.6f}, maxY={bounds_stac[3]:.6f}")
            download_bbox = bounds_stac
            download_cx = (bounds_stac[0] + bounds_stac[2]) / 2.0
            download_cy = (bounds_stac[1] + bounds_stac[3]) / 2.0
            print(f"    Centro: ({download_cx:.6f}, {download_cy:.6f})")
            print(f"  Comparação com polígono original:")
            print(f"    Polígono: X=[{bbox[0]:.6f}, {bbox[2]:.6f}] Y=[{bbox[1]:.6f}, {bbox[3]:.6f}]")
            print(f"    STAC:     X=[{bounds_stac[0]:.6f}, {bounds_stac[2]:.6f}] Y=[{bounds_stac[1]:.6f}, {bounds_stac[3]:.6f}]")
            dlon = abs(download_cx - cx)
            dlat = abs(download_cy - cy)
            print(f"    Diferença de centro: ΔLon={dlon:.6f}°, ΔLat={dlat:.6f}°")
        else:
            print("  ⚠ Footprint não disponível, usando área do polígono")
            download_bbox = bbox_square
            download_cx = sq_cx
            download_cy = sq_cy
            bounds_stac = None
        download_edge_px = edge_px
    else:
        print("\n📍 Modo POLÍGONO: usando quadrado ao redor do polígono...")
        download_bbox = bbox_square
        download_cx = sq_cx
        download_cy = sq_cy
        download_edge_px = edge_px
        bounds_stac = None  # Não há footprint
        print(f"  Quadrado: X=[{sq_min_lon:.6f}, {sq_max_lon:.6f}] Y=[{sq_min_lat:.6f}, {sq_max_lat:.6f}]")

    # 7. Download usando cubo (passando collection, start_date, end_date)
    print("\n⏳ Baixando todas as bandas (10m)...")
    da = cubo.create(
        collection=COLLECTION,
        start_date=selected_item.datetime.strftime('%Y-%m-%d'),
        end_date=selected_item.datetime.strftime('%Y-%m-%d'),
        bands=BANDAS_ALL,
        resolution=RESOLUTION,
        lat=download_cy,
        lon=download_cx,
        edge_size=download_edge_px
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
    cubo_clip = cubo_np.copy().astype("float32")
    if DOWNLOAD_FULL_TILE:
        print("Modo FULL TILE: mantendo imagem inteira sem recorte...")
    elif CROP_TO_TALHAO:
        print("Aplicando máscara do talhão ao raster final...")
        project = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
        geometria_proj = shapely_transform(project, geometria)
        mask = geometry_mask(
            [geometria_proj],
            transform=transform_cubo,
            invert=True,
            out_shape=(cubo_clip.shape[1], cubo_clip.shape[2])
        )
        for b in range(cubo_clip.shape[0]):
            cubo_clip[b][~mask] = np.nan
    else:
        print("Não aplicando máscara (mantendo dados do quadrado retangular)...")

    # 9. Salvar tudo
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Informações do raster final
    raster_height = cubo_clip.shape[1]
    raster_width = cubo_clip.shape[2]
    raster_size_m = raster_height * RESOLUTION  # em metros (quadrado)
    print(f"\n📊 RASTER FINAL:")
    print(f"  Dimensões: {raster_width}×{raster_height} pixels")
    print(f"  Tamanho em metros: {raster_size_m:.0f}m × {raster_size_m:.0f}m")
    print(f"  Extensão UTM (transform): {transform_cubo}")
    left = transform_cubo.c
    top = transform_cubo.f
    right = left + raster_width * transform_cubo.a
    bottom = top + raster_height * transform_cubo.e
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    min_lon, min_lat = transformer.transform(left, bottom)
    max_lon, max_lat = transformer.transform(right, top)
    print(f"  Extensão WGS84: X=[{min_lon:.6f}, {max_lon:.6f}] Y=[{min_lat:.6f}, {max_lat:.6f}]")
    print(f"  CRS: {crs}\n")
    
    # Comparação entre esperado e recebido
    print(f"📋 COMPARAÇÃO - ESPERADO vs RECEBIDO:")
    if DOWNLOAD_FULL_STAC_TILE and bounds_stac:
        print(f"  Modo: FULL STAC TILE")
        print(f"  Esperado (STAC): X=[{bounds_stac[0]:.6f}, {bounds_stac[2]:.6f}] Y=[{bounds_stac[1]:.6f}, {bounds_stac[3]:.6f}]")
        exp_bbox = bounds_stac
    else:
        print(f"  Modo: POLÍGONO")
        print(f"  Esperado (Polígono): X=[{bbox[0]:.6f}, {bbox[2]:.6f}] Y=[{bbox[1]:.6f}, {bbox[3]:.6f}]")
        exp_bbox = bbox
    print(f"  Recebido (Raster): X=[{min_lon:.6f}, {max_lon:.6f}] Y=[{min_lat:.6f}, {max_lat:.6f}]")
    
    # Converter para UTM para comparação em metros
    transformer_wgs_utm = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    exp_x0, exp_y0 = transformer_wgs_utm.transform(exp_bbox[0], exp_bbox[1])
    exp_x1, exp_y1 = transformer_wgs_utm.transform(exp_bbox[2], exp_bbox[3])
    
    rec_x0, rec_y0 = left, bottom
    rec_x1, rec_y1 = right, top
    
    print(f"  Esperado (UTM): X=[{exp_x0:.0f}, {exp_x1:.0f}]m Y=[{exp_y0:.0f}, {exp_y1:.0f}]m")
    print(f"  Recebido (UTM): X=[{rec_x0:.0f}, {rec_x1:.0f}]m Y=[{rec_y0:.0f}, {rec_y1:.0f}]m")
    
    dx = abs(rec_x0 - exp_x0)
    dy = abs(rec_y0 - exp_y0)
    print(f"  💡 Diferença de origem: ΔX={dx:.0f}m, ΔY={dy:.0f}m\n")

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