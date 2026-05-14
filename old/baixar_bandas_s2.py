import os
import sys
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.ops import unary_union

try:
    import geopandas as gpd
except ImportError:
    sys.exit("pip install geopandas")
try:
    from pystac_client import Client
except ImportError:
    sys.exit("pip install pystac-client")
try:
    import planetary_computer
    from planetary_computer import sign as pc_sign
except ImportError:
    sys.exit("pip install planetary-computer")

# Configurações
VETOR_LIMITE = "vetores/limite.gpkg"
START_DATE = "2024-09-08"
END_DATE = "2025-09-08"
COLECAO = "sentinel-2-l2a"
BANDAS = ["B04", "B03", "B02", "B08"]
RESOLUCAO_M = 10
OUTPUT_DIR = "resultados_rasters"
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

def list_and_select_item(bbox, start_date, end_date, collection=COLECAO):
    print("\nConsultando catálogo STAC...")
    catalog = Client.open(STAC_API_URL)
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 100}},
        max_items=100,
    )
    items = list(search.items())
    if not items:
        print("Nenhuma imagem encontrada para o período/área.")
        sys.exit(1)

    items.sort(key=lambda x: x.datetime)
    print(f"\n{'Índice':<6} {'Data':<21} {'Nuvem (%)':>10}")
    print("-" * 45)
    for i, item in enumerate(items):
        cloud = item.properties.get("eo:cloud_cover", "N/A")
        if isinstance(cloud, (int, float)):
            cloud_str = f"{cloud:.1f}"
        else:
            cloud_str = str(cloud)
        print(f"{i:<6} {str(item.datetime)[:19]:<21} {cloud_str:>10}")

    while True:
        choice = input("\nEscolha o índice da imagem para download (ou Enter para a primeira): ").strip()
        if choice == "":
            idx = 0
            break
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(items):
                break
        print(f"Entrada inválida. Digite um número entre 0 e {len(items)-1}, ou Enter para a primeira imagem.")

    selected = items[idx]
    print(f"\nImagem selecionada: {selected.id}")
    print(f"  Data: {selected.datetime}")
    print(f"  Nuvem: {selected.properties.get('eo:cloud_cover', 'N/A')}%\n")
    return selected

def download_bands(item, bandas, output_dir):
    # Assina os assets para Planetary Computer
    try:
        signed_item = pc_sign(item)
    except Exception as e:
        print(f"[!] Não foi possível assinar o item: {e}")
        signed_item = item

    assets = signed_item.assets
    os.makedirs(output_dir, exist_ok=True)
    for banda in bandas:
        asset_key = banda
        if asset_key not in assets:
            print(f"Banda {banda} não encontrada na cena {signed_item.id}.")
            continue
        url = assets[asset_key].href
        print(f"Baixando banda {banda} da cena {signed_item.id}...")
        try:
            with rasterio.open(url) as src:
                arr = src.read(1)
                profile = src.profile.copy()
                out_path = os.path.join(output_dir, f"{signed_item.id}_{banda}.tif")
                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(arr, 1)
            print(f"  Salvo em: {out_path}")
        except Exception as e:
            print(f"  Erro ao abrir banda {banda}: {e}")

def main():
    print("=" * 60)
    print("S2 - Download de Bandas (sem super-resolução)")
    print("=" * 60)

    # 1. Carregar vetor
    print(f"Lendo vetor: {VETOR_LIMITE}")
    gdf = gpd.read_file(VETOR_LIMITE)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    bbox = gdf.total_bounds  # [minx, miny, maxx, maxy]
    print(f"  Bbox: {bbox}\n")

    # 2. Listar imagens disponíveis e escolher uma cena
    selected_item = list_and_select_item(bbox, START_DATE, END_DATE)

    # 3. Baixar bandas da cena escolhida
    print(f"\nBaixando bandas da cena selecionada: {selected_item.id}")
    download_bands(selected_item, BANDAS, OUTPUT_DIR)

    print("\n[OK] As bandas selecionadas foram baixadas!")

if __name__ == "__main__":
    main()