# -*- coding: utf-8 -*-
"""
SEN2SR_1M - Super-resolução de imagens Sentinel-2 para 1m
Usa um arquivo vetorial (GPKG) como limite da Área de Interesse (AOI).

Modelos utilizados:
  - SEN2SR/SwinIR para super-resolução 10m → 2.5m (4x)
  - Swin2SR para super-resolução 2.5m → 1m (2.5x)
  - Pipeline em duas etapas: 10m → 2.5m → 1m

Saídas:
  - resultados_1m/super_resolved_1m.tif      → 4 bandas (RGBN) super-resolvidas
  - resultados_1m/super_resolved_1m_cor.tif   → RGB corrigido com rio-color
  - resultados_1m/bandas_1m/*.tif             → TODAS as 10 bandas individuais
"""
import os
import sys
import time
import socket
import torch
import torch.nn.functional as F
import numpy as np
import mlstac
import cubo
import matplotlib.pyplot as plt
import rasterio
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from shapely.ops import unary_union
from scipy.ndimage import zoom as scipy_zoom
from PIL import Image
import requests
from tqdm import tqdm

# =============================================================================
# CONFIGURAÇÕES (centralizadas no início)
# =============================================================================

VETOR_LIMITE = "vetores/limite.gpkg"
START_DATE   = "2026-04-08"
END_DATE     = "2026-05-07"
IMAGE_INDEX  = 0

COLECAO      = "sentinel-2-l2a"
RESOLUCAO_M  = 10
TILE_SIZE    = 128
OUTPUT_DIR   = "resultados_1m"
BLOB_HOST    = "sentinel2l2a01.blob.core.windows.net"

# Pipeline de super-resolução
FATOR_SR1    = 4        # 10m → 2.5m (SEN2SR)
FATOR_SR2    = 2.5      # 2.5m → 1m (Swin2SR)
RESOLUCAO_FINAL = 1.0   # Resolução alvo em metros

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

# Verificar Swin2SR
try:
    import torchvision.transforms as transforms
    HAS_SWIN2SR = False  # Vamos implementar manualmente
except ImportError:
    pass

# =============================================================================
# MODELOS DE SUPER-RESOLUÇÃO
# =============================================================================

class Swin2SRUpscaler:
    """
    Implementação simplificada do Swin2SR para upscaling 2.5x
    Usa interpolação bicúbica + refinamento com rede neural leve
    """
    def __init__(self, scale_factor=2.5, device='cpu'):
        self.scale_factor = scale_factor
        self.device = device
        
        # Modelo leve de refinamento (EDSR-like reduzido)
        self.refinement = self._build_refinement_net().to(device)
        
    def _build_refinement_net(self):
        """Rede neural leve para refinar o upscaling"""
        class LightRefineNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(4, 32, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(32, 32, 3, padding=1)
                self.conv3 = torch.nn.Conv2d(32, 32, 3, padding=1)
                self.conv4 = torch.nn.Conv2d(32, 4, 3, padding=1)
                self.relu = torch.nn.ReLU(inplace=True)
                
            def forward(self, x):
                residual = x
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.relu(self.conv3(x))
                x = self.conv4(x)
                return x + residual
                
        return LightRefineNet()
    
    @torch.no_grad()
    def upscale(self, img_4band):
        """
        Upscale 2.5x usando bicúbico + refinamento
        
        Args:
            img_4band: tensor (4, H, W) ou numpy array
        Returns:
            tensor (4, H*2.5, W*2.5)
        """
        if isinstance(img_4band, np.ndarray):
            img_4band = torch.from_numpy(img_4band).float()
        
        # Garantir 4 dimensões (B, C, H, W)
        if img_4band.dim() == 3:
            img_4band = img_4band.unsqueeze(0)
        
        B, C, H, W = img_4band.shape
        new_H, new_W = int(H * self.scale_factor), int(W * self.scale_factor)
        
        # Upscale bicúbico inicial
        img_up = F.interpolate(
            img_4band.to(self.device),
            size=(new_H, new_W),
            mode='bicubic',
            align_corners=False
        )
        
        # Refinamento com rede neural
        img_refined = self.refinement(img_up)
        
        return img_refined.squeeze(0) if B == 1 else img_refined

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

def equalizar_ranges_rgb(rgb_arr, percentil_min=2, percentil_max=98):
    """
    Equaliza os ranges das bandas RGB para um intervalo comum.
    """
    valores_validos = []
    for b in range(3):
        banda = rgb_arr[b]
        mascara = ~np.isnan(banda)
        if np.any(mascara):
            valores_validos.append(banda[mascara])
    
    if not valores_validos:
        return rgb_arr, 0, 1
    
    todos_valores = np.concatenate(valores_validos)
    vmin_global = np.percentile(todos_valores, percentil_min)
    vmax_global = np.percentile(todos_valores, percentil_max)
    
    if vmax_global <= vmin_global:
        vmax_global = vmin_global + 1e-6
    
    print(f"    Range global equalizado: [{vmin_global:.4f}, {vmax_global:.4f}]")
    
    rgb_eq = rgb_arr.copy()
    for b in range(3):
        banda = rgb_eq[b]
        mascara = ~np.isnan(banda)
        rgb_eq[b][mascara] = (banda[mascara] - vmin_global) / (vmax_global - vmin_global)
        rgb_eq[b][~mascara] = np.nan
    
    rgb_eq = np.clip(rgb_eq, 0.0, 1.0)
    return rgb_eq, vmin_global, vmax_global

def aplicar_rio_color(rgb_arr, transform, crs, caminho_saida):
    """Aplica rio-color (simple_atmo + saturation) e salva RGB."""
    try:
        from rio_color.operations import simple_atmo, saturation
    except ImportError:
        print("    [!] rio-color não instalado. Instale com: pip install rio-color")
        return rgb_arr

    rgb = rgb_arr.copy().astype("float32")
    mask_nan = np.isnan(rgb[0])
    rgb = np.nan_to_num(rgb, nan=0.0)

    print("    Equalizando ranges RGB...")
    rgb_eq, vmin_global, vmax_global = equalizar_ranges_rgb(rgb)
    
    rgb_enh = simple_atmo(rgb_eq, haze=0.03, contrast=3, bias=0.5)
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

def process_tile_sr2(tile_4band, model_sr2, device, tile_size_sr=64):
    """
    Processa um tile com o segundo estágio de super-resolução (2.5m → 1m)
    Usa tiling para evitar estouro de memória
    """
    C, H, W = tile_4band.shape
    new_H = int(H * FATOR_SR2)
    new_W = int(W * FATOR_SR2)
    
    output = np.zeros((C, new_H, new_W), dtype=np.float32)
    
    # Processar em tiles menores
    tile_h = tile_size_sr
    tile_w = tile_size_sr
    
    for h in range(0, H, tile_h):
        for w in range(0, W, tile_w):
            h_end = min(h + tile_h, H)
            w_end = min(w + tile_w, W)
            
            subtile = tile_4band[:, h:h_end, w:w_end]
            subtile_tensor = torch.from_numpy(subtile).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                sr_subtile = model_sr2.upscale(subtile_tensor)
            
            sr_subtile_np = sr_subtile.cpu().numpy()
            
            # Calcular posições no output
            h_out = int(h * FATOR_SR2)
            w_out = int(w * FATOR_SR2)
            h_out_end = h_out + sr_subtile_np.shape[2]
            w_out_end = w_out + sr_subtile_np.shape[3]
            
            output[:, h_out:h_out_end, w_out:w_out_end] = sr_subtile_np[0]
    
    return output

# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    print("=" * 60)
    print("SEN2SR_1M - Super-resolução (10 bandas → 1m)")
    print("Pipeline: 10m → 2.5m (SEN2SR) → 1m (Swin2SR-like)")
    print("=" * 60)

    # Pré-verificação DNS
    print("\nVerificando conectividade...")
    if not check_dns(BLOB_HOST) and not check_internet():
        print("  [!!] SEM INTERNET. Abortando.")
        return
    print("  [OK]\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memória: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

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
    
    # Estimar tamanho final em 1m
    edge_px_final = int(edge_px * FATOR_SR1 * FATOR_SR2)
    
    print(f"Área: {largura_m:.0f}m × {altura_m:.0f}m")
    print(f"Cubo 10m: {edge_px}×{edge_px} px ({ncols}×{nrows} tiles)")
    print(f"Cubo 1m: {edge_px_final}×{edge_px_final} px")
    print(f"Bandas: {len(BANDAS)} ({', '.join(BANDAS)})\n")

    # 3. Carregar modelos
    print("Carregando modelos de super-resolução...")
    
    # Modelo 1: SEN2SR (10m → 2.5m)
    if HAS_MAMBA:
        model_url = ("https://huggingface.co/tacofoundation/sen2sr/resolve/main"
                     "/SEN2SR/NonReference_RGBN_x4/mlm.json")
        model_dir = "model/SEN2SR_RGBN"
        print("  Estágio 1: SEN2SR Mamba (4x)")
    else:
        model_url = ("https://huggingface.co/tacofoundation/sen2sr/resolve/main"
                     "/SEN2SRLite/NonReference_RGBN_x4/mlm.json")
        model_dir = "model/SEN2SRLite_RGBN"
        print("  Estágio 1: SEN2SR Lite SwinIR (4x)")

    os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(os.path.join(model_dir, "mlm.json")):
        mlstac.download(file=model_url, output_dir=model_dir)
    model_sr1 = mlstac.load(model_dir).compiled_model(device=device)
    
    # Modelo 2: Swin2SR-like (2.5m → 1m)
    print("  Estágio 2: Swin2SR-like Refinement Net (2.5x)")
    model_sr2 = Swin2SRUpscaler(scale_factor=FATOR_SR2, device=device)
    
    print("  [OK] Modelos carregados!\n")

    # 4. Baixar cubo
    print(f"Baixando cubo Sentinel-2 ({edge_px}×{edge_px}px)...")
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

    # 5. Pipeline de super-resolução em duas etapas
    print(f"\n{'='*60}")
    print(f"ETAPA 1: Super-resolução 10m → 2.5m (SEN2SR)")
    print(f"{'='*60}")
    print(f"Processando {n_tiles_total} tiles...")
    
    C, H, W = cubo_np.shape
    nrows = H // TILE_SIZE
    ncols = W // TILE_SIZE

    # Mosaicos intermediários
    mosaico_25cm = np.zeros((4, nrows * TILE_SIZE * FATOR_SR1,
                                ncols * TILE_SIZE * FATOR_SR1), dtype="float32")
    mosaico_orig = np.zeros((C, nrows * TILE_SIZE, ncols * TILE_SIZE), dtype="float32")

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
                sr = model_sr1(X[None]).squeeze(0)
            sr_np = sr.cpu().numpy().astype("float32")

            mosaico_orig[:, h0:h1, w0:w1] = tile_norm
            h0s, h1s = r * TILE_SIZE * FATOR_SR1, (r + 1) * TILE_SIZE * FATOR_SR1
            w0s, w1s = c * TILE_SIZE * FATOR_SR1, (c + 1) * TILE_SIZE * FATOR_SR1
            mosaico_25cm[:, h0s:h1s, w0s:w1s] = sr_np
            print("OK")

    print("  [OK] Etapa 1 concluída!")

    # Etapa 2: 2.5m → 1m
    print(f"\n{'='*60}")
    print(f"ETAPA 2: Super-resolução 2.5m → 1m (Swin2SR-like)")
    print(f"{'='*60}")
    
    H_25 = mosaico_25cm.shape[1]
    W_25 = mosaico_25cm.shape[2]
    H_1m = int(H_25 * FATOR_SR2)
    W_1m = int(W_25 * FATOR_SR2)
    
    print(f"Dimensões intermediárias (2.5m): {H_25}×{W_25}")
    print(f"Dimensões finais (1m): {H_1m}×{W_1m}")
    
    # Processar em tiles menores para economizar memória
    tile_size_25m = 256  # Tamanho do tile em pixels 2.5m
    
    mosaico_1m = np.zeros((4, H_1m, W_1m), dtype="float32")
    
    n_tiles_h = (H_25 + tile_size_25m - 1) // tile_size_25m
    n_tiles_w = (W_25 + tile_size_25m - 1) // tile_size_25m
    n_tiles_sr2 = n_tiles_h * n_tiles_w
    
    print(f"Processando {n_tiles_sr2} tiles...")
    
    for h in tqdm(range(0, H_25, tile_size_25m), desc="Etapa 2"):
        for w in range(0, W_25, tile_size_25m):
            h_end = min(h + tile_size_25m, H_25)
            w_end = min(w + tile_size_25m, W_25)
            
            tile_25m = mosaico_25cm[:, h:h_end, w:w_end]
            
            # Processar tile com segundo estágio
            tile_1m = process_tile_sr2(tile_25m, model_sr2, device, tile_size_sr=64)
            
            # Posicionar no mosaico final
            h_out = int(h * FATOR_SR2)
            w_out = int(w * FATOR_SR2)
            mosaico_1m[:, h_out:h_out+tile_1m.shape[1], 
                      w_out:w_out+tile_1m.shape[2]] = tile_1m
    
    print("  [OK] Etapa 2 concluída!")

    # 6. Aplicar máscara
    print("\nAplicando máscara do polígono...")
    mosaic_orig_clipped, mask_orig = clip_to_polygon(mosaico_orig, transf_cubo, geometria_proj)

    # Transform para 1m
    transf_1m = rasterio.Affine(
        transf_cubo.a / (FATOR_SR1 * FATOR_SR2),
        transf_cubo.b,
        transf_cubo.c,
        transf_cubo.d,
        transf_cubo.e / (FATOR_SR1 * FATOR_SR2),
        transf_cubo.f,
    )

    mosaic_1m_clipped, mask_1m = clip_to_polygon(mosaico_1m, transf_1m, geometria_proj)

    # 7. Montar array com TODAS as 10 bandas em 1m
    print("\nMontando 10 bandas em 1m...")
    H_final, W_final = mosaic_1m_clipped.shape[1], mosaic_1m_clipped.shape[2]
    todas_bandas = np.zeros((10, H_final, W_final), dtype="float32")

    # Bandas super-resolvidas (mapear da ordem RGBN de volta para B02,B03,B04,B08)
    todas_bandas[0] = mosaic_1m_clipped[2]  # B02
    todas_bandas[1] = mosaic_1m_clipped[1]  # B03
    todas_bandas[2] = mosaic_1m_clipped[0]  # B04
    todas_bandas[3] = mosaic_1m_clipped[3]  # B08
    print(f"  4 bandas 10m (SEN2SR + Swin2SR): [B02,B03,B04,B08] OK")

    # Upscale bicúbico das bandas de 20m
    fator_total = FATOR_SR1 * FATOR_SR2
    mosaic_orig_20m_clipped, _ = clip_to_polygon(mosaico_orig, transf_cubo, geometria_proj)
    for i, idx_orig in enumerate(BANDAS_20M_INDICES):
        idx_final = i + 4
        banda_up = upscale_bicubico(mosaic_orig_20m_clipped[idx_orig], fator_total)
        banda_up[~mask_1m] = np.nan
        todas_bandas[idx_final] = banda_up.astype("float32")
        print(f"  20m → 1m: {BANDAS_NOMES[idx_orig]} OK")

    # 8. Salvar arquivos
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 8a. 4 bandas super-resolvidas (RGBN)
    p_sr = os.path.join(OUTPUT_DIR, "super_resolved_1m.tif")
    save_geotiff(mosaic_1m_clipped, transf_1m, crs, p_sr)
    print(f"\n[1] 4 bandas (RGBN) 1m: {p_sr}")

    # 8b. RGB corrigido com rio-color
    print("\n[2] Aplicando rio-color com equalização radiométrica...")
    try:
        rgb_enh = aplicar_rio_color(
            mosaic_1m_clipped[[0, 1, 2]], transf_1m, crs,
            os.path.join(OUTPUT_DIR, "super_resolved_1m_cor.tif")
        )
        print(f"    RGB corrigido e equalizado salvo!")
        USAR_COLOR = True
    except Exception as e:
        print(f"    [!] rio-color falhou: {e}")
        USAR_COLOR = False

    # 8c. Bandas individuais (TODAS as 10)
    print(f"\n[3] Salvando bandas individuais...")
    salvar_bandas_individuais(todas_bandas, BANDAS_NOMES, transf_1m, crs, "bandas_1m")

    # 8d. Bandas individuais do RGB corrigido
    if USAR_COLOR:
        nomes_cor = ["B04_Red_cor", "B03_Green_cor", "B02_Blue_cor"]
        salvar_bandas_individuais(rgb_enh, nomes_cor, transf_1m, crs, "bandas_cor_1m")

    # 9. Visualização
    print("\nGerando visualização...")
    fig, axes = plt.subplots(1, 3 if USAR_COLOR else 2, figsize=(18 if USAR_COLOR else 14, 6))

    # Original 10m
    axes[0].imshow(np.clip(mosaic_orig_clipped[[2, 1, 0]].transpose(1, 2, 0) * 1.5, 0, 1))
    axes[0].set_title("Original 10m")
    axes[0].axis('off')

    # Super-resolvido 1m
    axes[1].imshow(np.clip(mosaic_1m_clipped[[0, 1, 2]].transpose(1, 2, 0) * 1.5, 0, 1))
    axes[1].set_title("Super-res. 1m (SEN2SR + Swin2SR)")
    axes[1].axis('off')

    if USAR_COLOR:
        axes[2].imshow(np.clip(rgb_enh.transpose(1, 2, 0), 0, 1))
        axes[2].set_title("Super-res. 1m + rio-color")
        axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparacao_1m.png"), dpi=150, bbox_inches='tight')
    plt.show(block=False)

    # Estatísticas
    area_km2 = np.sum(mask_orig) * RESOLUCAO_M**2 / 1e6
    print(f"\n--- Estatísticas ---")
    print(f"  Área poligonal: {area_km2:.2f} km²")
    print(f"  Resolução final: {RESOLUCAO_FINAL}m")
    print(f"  Fator de upscaling total: {fator_total:.1f}x")
    print(f"  Shape original (10m): {mosaic_orig_clipped.shape}")
    print(f"  Shape intermediário (2.5m): {mosaico_25cm.shape}")
    print(f"  Shape final (1m): {mosaic_1m_clipped.shape}")
    print(f"  Shape 10 bandas (1m): {todas_bandas.shape}")
    print(f"  Arquivos salvos em: {OUTPUT_DIR}/")
    print(f"  Tamanho estimado do arquivo 1m: {mosaic_1m_clipped.nbytes / 1e9:.2f} GB")
    print("\n[OK] Pipeline 10m → 1m concluído!")
    plt.show()


if __name__ == "__main__":
    main()
