# -*- coding: utf-8 -*-
"""
SEN2SR - Super-resolução de imagens Sentinel-2 para 2.5m
Usa um arquivo vetorial (GPKG) como limite da Área de Interesse (AOI).

Pipeline:
  1. Lê o polígono do GPKG
  2. Calcula o cubo grande que cobre toda a bbox (edge_size múltiplo de 128)
  3. Fatura em tiles 128×128 (com padding se necessário)
  4. Aplica o modelo SEN2SR em cada tile
  5. Remonta o mosaico
  6. Recorta exatamente no formato do polígono
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

# =============================================================================
# CONFIGURAÇÕES (centralizadas no início)
# =============================================================================

VETOR_LIMITE = "vetores/limite.gpkg"
START_DATE   = "2024-09-08"
END_DATE     = "2025-09-08"
IMAGE_INDEX  = 0

COLECAO      = "sentinel-2-l2a"
BANDAS       = ["B04", "B03", "B02", "B08"]
RESOLUCAO_M  = 10
FATOR_SR     = 4        # 10m → 2.5m
TILE_SIZE    = 128
OUTPUT_DIR   = "resultados"
BLOB_HOST    = "sentinel2l2a01.blob.core.windows.net"

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
    """Adiciona padding ao array para que H e W sejam múltiplos de tile_size."""
    _, H, W = arr.shape
    pad_h = (tile_size - H % tile_size) % tile_size
    pad_w = (tile_size - W % tile_size) % tile_size
    if pad_h > 0 or pad_w > 0:
        arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
    return arr, pad_h, pad_w


def clip_to_polygon(arr, transform, geometria):
    """Aplica máscara: pixels fora do polígono viram NaN."""
    mask = geometry_mask([geometria], transform=transform, invert=True,
                         out_shape=(arr.shape[1], arr.shape[2]))
    masked = arr.copy().astype("float32")
    for b in range(arr.shape[0]):
        masked[b][~mask] = np.nan
    return masked, mask


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    print("=" * 60)
    print("SEN2SR - Super-resolução (Tiling adaptativo)")
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
    bbox = gdf.total_bounds  # [minx, miny, maxx, maxy]
    geometria = unary_union(gdf.geometry.values)
    print(f"  Bbox: {bbox}\n")

    # 2. Calcular edge_size para cobrir toda a bbox (múltiplo de TILE_SIZE)
    cy = (bbox[1] + bbox[3]) / 2.0
    cx = (bbox[0] + bbox[2]) / 2.0

    from math import cos, radians
    lat_rad = radians(cy)
    m_per_deg_lon = 111320.0 * cos(lat_rad)
    m_per_deg_lat = 111320.0

    largura_deg = bbox[2] - bbox[0]
    altura_deg  = bbox[3] - bbox[1]
    largura_m = largura_deg * m_per_deg_lon
    altura_m  = altura_deg  * m_per_deg_lat

    edge_px = int(max(np.ceil(largura_m / RESOLUCAO_M),
                      np.ceil(altura_m  / RESOLUCAO_M)))
    # Arredondar para múltiplo de TILE_SIZE
    edge_px = ((edge_px + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
    # Mínimo
    edge_px = max(edge_px, TILE_SIZE)

    ncols = edge_px // TILE_SIZE
    nrows = edge_px // TILE_SIZE
    print(f"Área: {largura_m:.0f}m × {altura_m:.0f}m")
    print(f"Cubo: {edge_px}×{edge_px} px ({ncols}×{nrows} tiles de {TILE_SIZE}px)\n")

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
    print(f"\nBaixando cubo S2 ({edge_px}×{edge_px}px, centro {cy:.6f}, {cx:.6f})...")
    da = cubo.create(lat=cy, lon=cx, collection=COLECAO, bands=BANDAS,
                     start_date=START_DATE, end_date=END_DATE,
                     edge_size=edge_px, resolution=RESOLUCAO_M)

    crs = da.rio.crs
    # Transform do cubo completo (rio já fornece o transform correto)
    transf_cubo = da.rio.transform()
    print(f"  CRS: {crs}")
    print(f"  Transform: {transf_cubo}")

    # CRS geralmente vem como None do cubo, mas os valores do transform
    # estão em coordenadas UTM (metros). Detectamos o EPSG pela localização.
    from rasterio.crs import CRS as RioCRS
    if crs is None:
        # Para a região de Palmas/TO (lat -10.18, lon -48.33), o UTM zone é 22S
        # Hemisfério sul = EPSG:32722
        crs = RioCRS.from_epsg(32722)
        print(f"  [~] CRS definido manualmente como: {crs}")

    # Reprojetar a geometria do polígono para o CRS do cubo (UTM)
    print("  [~] Reprojetando polígono para o CRS do cubo...")
    from shapely.ops import transform as shapely_transform
    import pyproj
    project = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    geometria_proj = shapely_transform(project, geometria)
    bbox_proj = geometria_proj.bounds  # [minx, miny, maxx, maxy] em UTM

    # Download
    cubo_np = compute_with_retry(da)
    print(f"  Shape: {cubo_np.shape}")

    # Ajustar padding se necessário (deve ser múltiplo, mas garantimos)
    cubo_np, pad_h, pad_w = pad_to_multiple(cubo_np, TILE_SIZE)
    if pad_h or pad_w:
        print(f"  Padding adicionado: {pad_h} linhas, {pad_w} colunas")
        # O transform não muda — os pixels adicionais ficam fora da área real

    # 5. Fatiar em tiles e processar
    C, H, W = cubo_np.shape
    nrows = H // TILE_SIZE
    ncols = W // TILE_SIZE
    n_tiles = nrows * ncols

    # Alocar mosaicos
    mosaico_orig = np.zeros((C, nrows * TILE_SIZE, ncols * TILE_SIZE), dtype="float32")
    mosaico_sr   = np.zeros((C, nrows * TILE_SIZE * FATOR_SR,
                                ncols * TILE_SIZE * FATOR_SR), dtype="float32")

    print(f"\nProcessando {n_tiles} tiles ({nrows}×{ncols})...")
    for r in range(nrows):
        for c in range(ncols):
            idx = r * ncols + c + 1
            h0, h1 = r * TILE_SIZE, (r + 1) * TILE_SIZE
            w0, w1 = c * TILE_SIZE, (c + 1) * TILE_SIZE
            print(f"  Tile {idx}/{n_tiles} [{r+1},{c+1}]...", end=" ")

            tile = cubo_np[:, h0:h1, w0:w1]
            tile_norm = (tile / 10000).astype("float32")

            X = torch.from_numpy(tile_norm).float().to(device)
            X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            with torch.no_grad():
                sr = model(X[None]).squeeze(0)
            sr_np = sr.cpu().numpy().astype("float32")

            # Inserir nos mosaicos
            mosaico_orig[:, h0:h1, w0:w1] = tile_norm
            h0s, h1s = r * TILE_SIZE * FATOR_SR, (r + 1) * TILE_SIZE * FATOR_SR
            w0s, w1s = c * TILE_SIZE * FATOR_SR, (c + 1) * TILE_SIZE * FATOR_SR
            mosaico_sr[:, h0s:h1s, w0s:w1s] = sr_np
            print("OK")

    print("  [OK] Todos os tiles processados!")

    # 6. Aplicar máscara do polígono
    print("\nAplicando máscara do polígono...")

    # O transform do cubo cobre edge_px x edge_px, centrado em (cx, cy).
    # O mosaico original tem (nrows*TILE_SIZE) x (ncols*TILE_SIZE) = edge_px x edge_px.
    # Logo o transform original já está correto para o mosaico.
    # O canto superior esquerdo do cubo (pixel 0,0) está em:
    #   transf_cubo * (0, 0) -> (c, f)
    # Isso já é o que precisamos.

    mosaic_orig_clipped, mask_orig = clip_to_polygon(mosaico_orig, transf_cubo, geometria_proj)

    # Transform para o super-resolvido (resolução 4x maior)
    # O FATOR_SR multiplica as dimensões, então os pixels são FATOR_SR vezes menores.
    transf_sr = rasterio.Affine(
        transf_cubo.a / FATOR_SR,  # resolução x dividida pelo fator
        transf_cubo.b,
        transf_cubo.c,
        transf_cubo.d,
        transf_cubo.e / FATOR_SR,  # resolução y dividida pelo fator
        transf_cubo.f,
    )

    mosaic_sr_clipped, mask_sr = clip_to_polygon(mosaico_sr, transf_sr, geometria_proj)

    # 7. Salvar GeoTIFFs originais
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    p_orig = os.path.join(OUTPUT_DIR, "original_10m_mosaic.tif")
    save_geotiff(mosaic_orig_clipped, transf_cubo, crs, p_orig)
    print(f"Original: {p_orig}")

    p_sr = os.path.join(OUTPUT_DIR, "super_resolved_2_5m_mosaic.tif")
    save_geotiff(mosaic_sr_clipped, transf_sr, crs, p_sr)
    print(f"Super-res.: {p_sr}")

    # 7.5 Aplicar rio-color (melhoria de cores) no super-resolvido
    try:
        from rio_color.operations import simple_atmo, saturation
        import rio_color.utils as rio_utils
        print("\nAplicando rio-color (correção atmosférica + saturação)...")

        # Extrair apenas as bandas RGB (B04, B03, B02)
        rgb_sr = mosaic_sr_clipped[[0, 1, 2]].copy().astype("float32")

        # Guardar máscara de NaN para restaurar depois
        mask_nan = np.isnan(rgb_sr[0])

        # Substituir NaN por 0 para processamento e normalizar para [0,1]
        rgb_sr = np.nan_to_num(rgb_sr, nan=0.0)
        
        # Normalizar para [0,1] - o modelo pode gerar valores fora desta faixa
        # Primeiro clamps para [0, max] depois divide pelo max global
        vmin, vmax = rgb_sr.min(), rgb_sr.max()
        if vmax > 1.0 or vmin < 0.0:
            if vmax - vmin > 0.001:
                rgb_norm = (rgb_sr - vmin) / (vmax - vmin)
            else:
                rgb_norm = np.clip(rgb_sr, 0, 1)
        else:
            rgb_norm = rgb_sr

        # simple_atmo: correção atmosférica visual
        #   haze=0.03 (névoa leve), contrast=3 (típico), bias=0.5 (centro)
        rgb_enhanced = simple_atmo(rgb_norm, haze=0.03, contrast=3, bias=0.5)

        # saturation: aumentar saturação em 30%
        rgb_enhanced = saturation(rgb_enhanced, proportion=1.3)

        # Desnormalizar de volta para a escala original (opcional)
        # Mantemos em [0,1] porque é o range correto para visualização
        rgb_enhanced = np.clip(rgb_enhanced, 0.0, 1.0)

        # Restaurar NaN
        for b in range(3):
            rgb_enhanced[b][mask_nan] = np.nan

        # Salvar RGB com cores melhoradas (3 bandas, float32 em [0,1])
        p_rgb = os.path.join(OUTPUT_DIR, "super_resolved_2_5m_cor.tif")
        with rasterio.open(p_rgb, 'w', driver='GTiff',
                           height=rgb_enhanced.shape[1], width=rgb_enhanced.shape[2],
                           count=3, dtype=rasterio.float32,
                           crs=crs, transform=transf_sr, compress='lzw') as dst:
            dst.write(rgb_enhanced.astype(rasterio.float32))
        print(f"Super-res. corrigido: {p_rgb}")

        # Criar versão 4-bandas (substitui RGB, mantém NIR)
        mosaic_sr_color = mosaic_sr_clipped.copy()
        mosaic_sr_color[[0, 1, 2]] = rgb_enhanced
        p_sr_color = os.path.join(OUTPUT_DIR, "super_resolved_2_5m_color.tif")
        save_geotiff(mosaic_sr_color, transf_sr, crs, p_sr_color)
        print(f"Super-res. (4 bandas): {p_sr_color}")

        USAR_COLOR = True
    except Exception as e:
        print(f"  [!] Erro ao aplicar rio-color: {e}")
        import traceback
        traceback.print_exc()
        print("  [!] (rio-color está instalado mas houve erro nos dados)")
        USAR_COLOR = False
        mosaic_sr_color = mosaic_sr_clipped

    # 8. Visualização
    print("\nGerando visualização...")
    n_panels = 3 if USAR_COLOR else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 7))

    axes[0].imshow(np.clip(mosaic_orig_clipped[[0, 1, 2]].transpose(1, 2, 0) * 1.5, 0, 1))
    axes[0].set_title(f"Original 10m ({nrows}×{ncols} tiles)")
    axes[0].axis('off')

    axes[1].imshow(np.clip(mosaic_sr_clipped[[0, 1, 2]].transpose(1, 2, 0) * 1.5, 0, 1))
    axes[1].set_title(f"Super-res. 2.5m ({FATOR_SR}×)")
    axes[1].axis('off')

    if USAR_COLOR:
        axes[2].imshow(np.clip(mosaic_sr_color[[0, 1, 2]].transpose(1, 2, 0), 0, 1))
        axes[2].set_title(f"Super-res. + rio-color")
        axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparacao.png"), dpi=150, bbox_inches='tight')
    print(f"Comparação: {OUTPUT_DIR}/comparacao.png")

    # Estatísticas
    area_km2_orig = np.sum(mask_orig) * RESOLUCAO_M**2 / 1e6
    area_km2_sr   = np.sum(mask_sr) * (RESOLUCAO_M / FATOR_SR)**2 / 1e6
    print(f"\n  Área poligonal: {area_km2_orig:.2f} km² (orig), {area_km2_sr:.2f} km² (sr)")
    print(f"  Shape orig: {mosaic_orig_clipped.shape}")
    print(f"  Shape sr:   {mosaic_sr_clipped.shape}")

    plt.show()
    print("\n[OK] Concluído!")


if __name__ == "__main__":
    main()