"""
================================================================================
  SENTINEL-2 DOWNLOADER INTERATIVO
  Baixa imagens Sentinel-2 L2A via STAC API (Element 84 / AWS)
  Compatível com Windows | Python 3.8+
================================================================================
  DEPENDÊNCIAS:
    pip install pystac-client rasterio geopandas numpy requests tqdm pyproj shapely

  USO:
    Edite o bloco CONFIGURAÇÕES abaixo e execute:
    python sentinel2_downloader.py
================================================================================
"""

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURAÇÕES DO USUÁRIO                           ║
# ║              Edite apenas este bloco antes de executar                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from datetime import datetime as _dt

# ── Área e período ────────────────────────────────────────────────────────────
SHAPEFILE    = r"vetores/limite.gpkg"     # Caminho para o vetor (shp/gpkg/geojson)
DATA_INICIO  = f"{_dt.now().year}-01-01"  # Início: usa o ano corrente automaticamente
DATA_FIM     = f"{_dt.now().year}-12-31"  # Fim: usa o ano corrente automaticamente
MAX_NUVENS   = 50.0                       # Cobertura máxima de nuvens (%)

# ── Seleção de imagens ────────────────────────────────────────────────────────
# Números das imagens da tabela exibida, separados por vírgula. Ex: "1,3,5"
# Use "1-5" para intervalo, ou "all" para todas.
IMAGENS_SELECIONADAS = "all"

# ── Bandas para download ──────────────────────────────────────────────────────
# Opções prontas (escolha UMA):
#   "rgb"       → B04, B03, B02  (Cor natural)
#   "falsa_cor" → B08, B04, B03  (Falsa Cor NIR)
#   "swir"      → B12, B8A, B04  (Vegetação SWIR)
#   "agri"      → B11, B08, B02  (Agricultura)
#   "urbano"    → B12, B11, B04  (Índice Urbano)
#   "todas"     → todas as bandas disponíveis
# Ou liste manualmente as bandas que quiser:
#   BANDAS = ["B04", "B03", "B02", "B08"]
BANDAS = "rgb"

# ── Recorte pela máscara do shapefile ─────────────────────────────────────────
# True  → recortar pelo polígono exato do shapefile
# False → baixar a cena inteira (~110×110 km)
RECORTAR = True

# ── Projeção de saída ─────────────────────────────────────────────────────────
# "shapefile" → usa o EPSG do shapefile de entrada
# "utm"       → mantém a projeção UTM nativa do Sentinel-2
# número      → qualquer EPSG personalizado, ex: 32722 ou 4674
EPSG_SAIDA = "utm"

# ── Pasta de saída ────────────────────────────────────────────────────────────
PASTA_SAIDA = r"sentinel2_downloads"

# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import json
import time
import warnings
import traceback
from pathlib import Path

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
#  VERIFICAÇÃO E INSTALAÇÃO DE DEPENDÊNCIAS
# ──────────────────────────────────────────────
def verificar_dependencias():
    pacotes = {
        "pystac_client": "pystac-client",
        "rasterio": "rasterio",
        "geopandas": "geopandas",
        "numpy": "numpy",
        "requests": "requests",
        "tqdm": "tqdm",
        "pyproj": "pyproj",
        "shapely": "shapely",
    }
    faltando = []
    for modulo, pip_nome in pacotes.items():
        try:
            __import__(modulo)
        except ImportError:
            faltando.append(pip_nome)

    if faltando:
        print("\n[AVISO] Pacotes ausentes detectados:")
        for p in faltando:
            print(f"  - {p}")
        resposta = input("\nDeseja instalar automaticamente agora? (s/n): ").strip().lower()
        if resposta == "s":
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + faltando)
            print("\n[OK] Pacotes instalados. Reiniciando importações...\n")
        else:
            print("\nInstale manualmente com:\n  pip install " + " ".join(faltando))
            sys.exit(1)

verificar_dependencias()

# ──────────────────────────────────────────────
#  IMPORTAÇÕES PRINCIPAIS
# ──────────────────────────────────────────────
import numpy as np
import requests
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS, Transformer
from shapely.geometry import shape, mapping, box
from shapely.ops import transform as shapely_transform
import functools
from tqdm import tqdm
from pystac_client import Client

# ──────────────────────────────────────────────
#  CONSTANTES
# ──────────────────────────────────────────────
STAC_URL     = "https://earth-search.aws.element84.com/v1"
COLLECTION   = "sentinel-2-l2a"
RESOLUCAO_M  = 10  # metros por pixel

# Mapa: nome do asset no STAC Element84 → código de banda Sentinel-2
# O STAC usa nomes descritivos ("red", "blue"...) em vez de "B04", "B02"...
ASSET_MAP = {
    "B01": ["coastal",   "B01"],
    "B02": ["blue",      "B02"],
    "B03": ["green",     "B03"],
    "B04": ["red",       "B04"],
    "B05": ["rededge1",  "B05"],
    "B06": ["rededge2",  "B06"],
    "B07": ["rededge3",  "B07"],
    "B08": ["nir",       "B08"],
    "B8A": ["nir08",     "B8A"],
    "B09": ["nir09",     "B09"],
    "B11": ["swir16",    "B11"],
    "B12": ["swir22",    "B12"],
    "SCL": ["scl",       "SCL"],
}

BANDAS_DISPONIVEIS = {
    "B01": ("Coastal Aerosol",         "60m"),
    "B02": ("Blue",                    "10m"),
    "B03": ("Green",                   "10m"),
    "B04": ("Red",                     "10m"),
    "B05": ("Red Edge",                "20m"),
    "B06": ("Red Edge",                "20m"),
    "B07": ("Red Edge",                "20m"),
    "B08": ("Near Infrared",           "10m"),
    "B8A": ("Red Edge",                "20m"),
    "B09": ("Water Vapour",            "60m"),
    "B11": ("SWIR 1",                  "20m"),
    "B12": ("SWIR 2",                  "20m"),
    "SCL": ("Scene Classification",    "20m"),
}

COMPOSICOES_PRONTAS = {
    "1": {"nome": "RGB Natural",          "bandas": ["B04", "B03", "B02"]},
    "2": {"nome": "Falsa Cor (NIR)",      "bandas": ["B08", "B04", "B03"]},
    "3": {"nome": "SWIR (Vegetação)",     "bandas": ["B12", "B8A", "B04"]},
    "4": {"nome": "Agricultura",          "bandas": ["B11", "B08", "B02"]},
    "5": {"nome": "Índice Urbano",        "bandas": ["B12", "B11", "B04"]},
    "6": {"nome": "Todas as bandas",      "bandas": list(BANDAS_DISPONIVEIS.keys())},
    "7": {"nome": "Escolher manualmente", "bandas": None},
}

SEP = "=" * 72


# ══════════════════════════════════════════════════════════════════════════════
#  FUNÇÕES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def cabecalho():
    print(f"\n{SEP}")
    print("   SENTINEL-2 DOWNLOADER INTERATIVO")
    print("   STAC: Element 84 / AWS | Coleção: sentinel-2-l2a")
    print(SEP)


def secao(titulo: str):
    print(f"\n{'─' * 72}")
    print(f"  {titulo}")
    print(f"{'─' * 72}")


def ler_shapefile(caminho: str) -> gpd.GeoDataFrame:
    """Lê qualquer shapefile/GeoJSON/GPKG e reprojeta para WGS84 para a busca STAC."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    gdf = gpd.read_file(caminho)
    epsg_original = gdf.crs.to_epsg() if gdf.crs else None

    print(f"\n  [Shape] {caminho.name}")
    print(f"  [CRS original] EPSG:{epsg_original} — {gdf.crs.name if gdf.crs else 'desconhecido'}")
    print(f"  [Geometrias] {len(gdf)} feições | Tipo: {gdf.geom_type.iloc[0]}")

    # Reprojetar para WGS84 (necessário para STAC bbox)
    if epsg_original != 4326:
        print(f"  [Info] Reprojetando para WGS84 (EPSG:4326) para busca STAC...")
        gdf_wgs84 = gdf.to_crs(epsg=4326)
    else:
        gdf_wgs84 = gdf

    return gdf, gdf_wgs84, epsg_original


def bbox_wgs84(gdf_wgs84: gpd.GeoDataFrame) -> list:
    """Retorna [minx, miny, maxx, maxy] em WGS84."""
    bounds = gdf_wgs84.total_bounds
    return [round(float(v), 6) for v in bounds]


def buscar_imagens(bbox: list, data_inicio: str, data_fim: str, max_nuvens: float = 100.0) -> list:
    """Busca itens STAC na área e período informados."""
    print(f"\n  [STAC] Conectando em {STAC_URL}...")
    catalog = Client.open(STAC_URL)

    print(f"  [Busca] Área: {bbox}")
    print(f"  [Busca] Período: {data_inicio} → {data_fim}")
    print(f"  [Busca] Nuvens máximas: {max_nuvens}%")

    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{data_inicio}/{data_fim}",
        query={"eo:cloud_cover": {"lte": max_nuvens}},
        max_items=200,
    )

    itens = list(search.items())
    # Ordenar por data localmente (sortby remoto não é suportado neste índice)
    itens.sort(key=lambda i: i.properties.get("datetime", ""))
    return itens


def exibir_tabela_imagens(itens: list) -> None:
    """Exibe tabela formatada com data, tile, nuvens e plataforma."""
    secao("IMAGENS DISPONÍVEIS")
    print(f"\n  {'#':>3}  {'DATA':^12}  {'TILE':^10}  {'NUVENS':>7}  {'PLATAFORMA':^12}  {'ID STAC'}")
    print(f"  {'─'*3}  {'─'*12}  {'─'*10}  {'─'*7}  {'─'*12}  {'─'*40}")

    for i, item in enumerate(itens, 1):
        props   = item.properties
        data    = props.get("datetime", "?")[:10]
        nuvens  = props.get("eo:cloud_cover", -1)
        tile    = props.get("mgrs:utm_zone", "?")
        # tile completo
        tile_id = props.get("s2:mgrs_tile", tile)
        plat    = props.get("platform", "?").upper()
        stac_id = item.id[:40]

        # Emoji de qualidade
        if nuvens < 10:
            qualidade = "🟢"
        elif nuvens < 30:
            qualidade = "🟡"
        elif nuvens < 60:
            qualidade = "🟠"
        else:
            qualidade = "🔴"

        print(f"  {i:>3}  {data:^12}  {tile_id:^10}  {nuvens:>6.1f}% {qualidade}  {plat:^12}  {stac_id}")

    print(f"\n  Total: {len(itens)} imagens encontradas")
    print("  🟢 <10%  🟡 10-30%  🟠 30-60%  🔴 >60% cobertura de nuvens\n")




# ══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD E PROCESSAMENTO
# ══════════════════════════════════════════════════════════════════════════════

def baixar_url(url: str, destino: Path, desc: str = "") -> bool:
    """Faz download de uma URL com barra de progresso."""
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(destino, "wb") as f, tqdm(
            desc=f"    {desc}",
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            ncols=70,
            leave=False,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [Erro] Download falhou: {e}")
        return False


def reprojetar_raster(src_path: Path, dst_path: Path, epsg_destino: int):
    """Reprojeta um GeoTIFF para o EPSG desejado."""
    with rasterio.open(src_path) as src:
        if src.crs.to_epsg() == epsg_destino:
            return  # já está na projeção correta

        transform, width, height = calculate_default_transform(
            src.crs, CRS.from_epsg(epsg_destino), src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": CRS.from_epsg(epsg_destino),
            "transform": transform,
            "width": width,
            "height": height,
        })
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=CRS.from_epsg(epsg_destino),
                    resampling=Resampling.bilinear,
                )


def recortar_raster(src_path: Path, dst_path: Path, geometrias_wgs84, epsg_saida: int):
    """Recorta um GeoTIFF pelas geometrias do shapefile."""
    with rasterio.open(src_path) as src:
        crs_raster = src.crs

        # Reprojetar geometria para o CRS do raster
        if crs_raster.to_epsg() != 4326:
            transformer = Transformer.from_crs("EPSG:4326", crs_raster.to_string(), always_xy=True)
            geoms_reproj = [
                mapping(shapely_transform(
                    functools.partial(transformer.transform),
                    shape(g)
                ))
                for g in geometrias_wgs84
            ]
        else:
            geoms_reproj = geometrias_wgs84

        out_image, out_transform = rio_mask(src, geoms_reproj, crop=True, nodata=0)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width":  out_image.shape[2],
            "transform": out_transform,
        })

    with rasterio.open(dst_path, "w", **out_meta) as dst:
        dst.write(out_image)

    # Reprojetar para EPSG de saída se necessário
    if epsg_saida and crs_raster.to_epsg() != epsg_saida:
        tmp = dst_path.with_suffix(".tmp.tif")
        dst_path.rename(tmp)
        reprojetar_raster(tmp, dst_path, epsg_saida)
        tmp.unlink()


def processar_item(item, bandas: list, gdf_wgs84: gpd.GeoDataFrame,
                   recortar: bool, epsg_saida, pasta: Path):
    """Baixa e processa um item STAC (uma cena)."""
    props   = item.properties
    assets  = item.assets

    # ── Extrair metadados do item ──────────────────────────────────────────────
    # ID exemplo: S2C_23MKR_20260121_0_L2A  ou  S2B_22MHA_20260322_0_L2A
    partes  = item.id.split("_")
    plat    = partes[0] if len(partes) > 0 else props.get("platform", "S2x")
    tile_id = partes[1] if len(partes) > 1 else props.get("s2:mgrs_tile", "XX")
    data    = props.get("datetime", "")[:10].replace("-", "")
    prefixo = f"{data}_{tile_id}_{plat}"

    pasta_item = pasta / prefixo
    pasta_item.mkdir(parents=True, exist_ok=True)

    print(f"\n  📥 Processando: {prefixo}")
    print(f"     Nuvens: {props.get('eo:cloud_cover', '?'):.1f}%")
    print(f"     Assets disponíveis: {list(assets.keys())}")

    geoms      = [mapping(g) for g in gdf_wgs84.geometry]
    resultados = []

    for banda in bandas:
        # Buscar o asset pelo mapa de nomes alternativos
        chaves_possiveis = ASSET_MAP.get(banda, [banda])
        asset_key = None
        for chave in chaves_possiveis:
            if chave in assets:
                asset_key = chave
                break

        if asset_key is None:
            print(f"     [Aviso] Banda {banda} não encontrada nos assets desta cena.")
            continue

        url           = assets[asset_key].href
        nome_arquivo  = f"{prefixo}_{banda}.tif"
        caminho_temp  = pasta_item / f"_tmp_{banda}.tif"
        caminho_final = pasta_item / nome_arquivo

        if caminho_final.exists():
            print(f"     [Pulando] {nome_arquivo} já existe.")
            resultados.append(caminho_final)
            continue

        desc_banda = BANDAS_DISPONIVEIS.get(banda, ("?", "?"))[0]
        print(f"     ↓ {banda} — {desc_banda} (asset: '{asset_key}')...")

        ok = baixar_url(url, caminho_temp, desc=banda)
        if not ok:
            continue

        try:
            if recortar:
                epsg_out = epsg_saida
                if epsg_out is None:
                    with rasterio.open(caminho_temp) as src:
                        epsg_out = src.crs.to_epsg()
                recortar_raster(caminho_temp, caminho_final, geoms, epsg_out)
                caminho_temp.unlink(missing_ok=True)
            else:
                epsg_out = epsg_saida
                if epsg_out is None:
                    with rasterio.open(caminho_temp) as src:
                        epsg_out = src.crs.to_epsg()

                if epsg_out:
                    reprojetar_raster(caminho_temp, caminho_final, epsg_out)
                    caminho_temp.unlink(missing_ok=True)
                else:
                    caminho_temp.rename(caminho_final)

            print(f"       ✅ Salvo: {caminho_final.name}")
            resultados.append(caminho_final)

        except Exception as e:
            print(f"       ❌ Erro ao processar {banda}: {e}")
            traceback.print_exc()
            caminho_temp.unlink(missing_ok=True)

    # Salvar metadados JSON da cena
    meta_path = pasta_item / f"{prefixo}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "id":              item.id,
            "data":            props.get("datetime"),
            "tile":            tile_id,
            "plataforma":      plat,
            "nuvens_%":        props.get("eo:cloud_cover"),
            "epsg_original":   props.get("proj:epsg"),
            "bandas_baixadas": [str(r.name) for r in resultados],
            "recortado":       recortar,
            "epsg_saida":      epsg_saida,
        }, f, indent=2, ensure_ascii=False)

    print(f"     📄 Metadados: {meta_path.name}")
    return resultados


# ══════════════════════════════════════════════════════════════════════════════
#  FLUXO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def resolver_bandas(bandas_cfg) -> list:
    """Traduz a variável BANDAS (string ou lista) para lista de códigos."""
    PRESETS = {
        "rgb":       ["B04", "B03", "B02"],
        "falsa_cor": ["B08", "B04", "B03"],
        "swir":      ["B12", "B8A", "B04"],
        "agri":      ["B11", "B08", "B02"],
        "urbano":    ["B12", "B11", "B04"],
        "todas":     list(BANDAS_DISPONIVEIS.keys()),
    }
    if isinstance(bandas_cfg, list):
        return bandas_cfg
    chave = bandas_cfg.strip().lower()
    if chave in PRESETS:
        return PRESETS[chave]
    # Tratado como lista separada por vírgula
    return [b.strip().upper() for b in bandas_cfg.split(",") if b.strip()]


def resolver_epsg(epsg_cfg, epsg_shapefile: int):
    """Traduz a variável EPSG_SAIDA para int ou None (UTM nativo)."""
    if isinstance(epsg_cfg, int):
        return epsg_cfg
    v = str(epsg_cfg).strip().lower()
    if v == "shapefile":
        return epsg_shapefile
    if v == "utm":
        return None  # mantém UTM nativo da cena
    try:
        return int(v)
    except ValueError:
        raise ValueError(f"EPSG_SAIDA inválido: '{epsg_cfg}'. Use 'shapefile', 'utm' ou um número EPSG.")


def resolver_selecao(selecao_cfg: str, itens: list) -> list:
    """Traduz IMAGENS_SELECIONADAS para lista de items."""
    v = selecao_cfg.strip().lower()
    if v == "all":
        return itens

    selecionados = []
    for parte in v.split(","):
        parte = parte.strip()
        if "-" in parte:
            ini, fim = parte.split("-")
            for n in range(int(ini), int(fim) + 1):
                if 1 <= n <= len(itens):
                    selecionados.append(itens[n - 1])
        else:
            n = int(parte)
            if 1 <= n <= len(itens):
                selecionados.append(itens[n - 1])

    # Remover duplicatas mantendo ordem
    vistos, unicos = set(), []
    for item in selecionados:
        if item.id not in vistos:
            vistos.add(item.id)
            unicos.append(item)
    return unicos


def main():
    cabecalho()

    # ── Resolver configurações ─────────────────────────────────────────────────
    bandas   = resolver_bandas(BANDAS)
    recortar = bool(RECORTAR)

    print(f"\n  Shapefile  : {SHAPEFILE}")
    print(f"  Período    : {DATA_INICIO} → {DATA_FIM}")
    print(f"  Nuvens máx.: {MAX_NUVENS}%")
    print(f"  Bandas     : {bandas}")
    print(f"  Recortar   : {'Sim' if recortar else 'Não'}")
    print(f"  EPSG saída : {EPSG_SAIDA}")
    print(f"  Saída      : {PASTA_SAIDA}")

    # ── 1. Ler shapefile ───────────────────────────────────────────────────────
    secao("1. LENDO SHAPEFILE")
    try:
        gdf_original, gdf_wgs84, epsg_original = ler_shapefile(SHAPEFILE)
    except FileNotFoundError as e:
        print(f"\n  [ERRO] {e}")
        sys.exit(1)

    epsg_saida = resolver_epsg(EPSG_SAIDA, epsg_original)
    epsg_label = epsg_saida if epsg_saida else "Nativo da cena (UTM)"
    print(f"  [Info] EPSG de saída resolvido: {epsg_label}")

    # ── 2. Buscar imagens no STAC ──────────────────────────────────────────────
    secao("2. BUSCANDO IMAGENS NO STAC")
    bbox = bbox_wgs84(gdf_wgs84)

    try:
        itens = buscar_imagens(bbox, DATA_INICIO, DATA_FIM, MAX_NUVENS)
    except Exception as e:
        print(f"\n  [ERRO] Falha na busca STAC: {e}")
        traceback.print_exc()
        sys.exit(1)

    if not itens:
        print(f"\n  [Aviso] Nenhuma imagem encontrada para os critérios informados.")
        print(f"  Sugestões: ampliar período ou aumentar MAX_NUVENS.")
        sys.exit(0)

    # ── 3. Exibir tabela ───────────────────────────────────────────────────────
    exibir_tabela_imagens(itens)

    # ── 4. Resolver seleção de imagens ─────────────────────────────────────────
    try:
        itens_selecionados = resolver_selecao(IMAGENS_SELECIONADAS, itens)
    except Exception as e:
        print(f"\n  [ERRO] IMAGENS_SELECIONADAS inválido: {e}")
        sys.exit(1)

    if not itens_selecionados:
        print("\n  [Aviso] Nenhuma imagem válida na seleção. Verifique IMAGENS_SELECIONADAS.")
        sys.exit(0)

    # ── 5. Resumo ──────────────────────────────────────────────────────────────
    secao("RESUMO DO DOWNLOAD")
    print(f"\n  Imagens selecionadas : {len(itens_selecionados)}")
    print(f"  Bandas               : {bandas}")
    print(f"  Recortar pela máscara: {'Sim' if recortar else 'Não'}")
    print(f"  EPSG de saída        : {epsg_label}")
    print(f"  Pasta de saída       : {PASTA_SAIDA}")

    # ── 6. Pasta de saída ──────────────────────────────────────────────────────
    pasta = Path(PASTA_SAIDA)
    pasta.mkdir(parents=True, exist_ok=True)
    print(f"\n  [OK] Arquivos serão salvos em: {pasta.resolve()}")

    # ── 7. Processar cada cena ─────────────────────────────────────────────────
    secao("DOWNLOADING")
    total_arquivos = []
    t0 = time.time()

    for idx, item in enumerate(itens_selecionados, 1):
        print(f"\n  [{idx}/{len(itens_selecionados)}]", end="")
        arquivos = processar_item(item, bandas, gdf_wgs84, recortar, epsg_saida, pasta)
        total_arquivos.extend(arquivos)

    # ── 8. Relatório final ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    secao("CONCLUÍDO")
    print(f"\n  ✅ {len(total_arquivos)} arquivo(s) baixado(s) com sucesso")
    print(f"  ⏱  Tempo total: {elapsed/60:.1f} min")
    print(f"  📁 Local      : {pasta.resolve()}")
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()