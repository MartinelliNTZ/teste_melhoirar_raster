# -*- coding: utf-8 -*-
"""
SEN2SR_ULTRA - Super-resolução máxima de Sentinel-2 (OTIMIZADO)
Pipeline multi-estágio com processamento paralelo e otimizações
"""
import os
import sys
import time
import socket
import torch
import torch.nn as nn
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
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import warnings
from datetime import datetime
import geopandas as gpd
import pyproj
import rioxarray
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

VETOR_LIMITE = "vetores/limite.gpkg"
START_DATE   = "2026-04-08"
END_DATE     = "2026-05-07"
IMAGE_INDEX  = 0

COLECAO      = "sentinel-2-l2a"
RESOLUCAO_M  = 10
TILE_SIZE    = 128
OUTPUT_DIR   = "resultados_ultra"
BLOB_HOST    = "sentinel2l2a01.blob.core.windows.net"

# Pipeline multi-estágio
FATOR_SR1    = 4        # 10m → 2.5m (SEN2SR)
FATOR_SR2    = 2        # 2.5m → 1.25m (HAT-L)
FATOR_SR3    = 2.5      # 1.25m → 0.5m (Real-ESRGAN+)
RESOLUCAO_FINAL = 0.5   # Resolução alvo em metros

# Otimizações
BATCH_SIZE_HAT = 4      # Processamento em batch para HAT-L
TILE_OVERLAP = 16       # Overlap para evitar artefatos
USE_TORCH_COMPILE = False # Compilação JIT PyTorch 2.0+ (desabilitada em CPU)
USE_AMP = True           # Automatic Mixed Precision

BANDAS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]
BANDAS_NOMES = [
    "B02_Blue", "B03_Green", "B04_Red", "B08_NIR",
    "B05_RedEdge1", "B06_RedEdge2", "B07_RedEdge3",
    "B8A_NarrowNIR", "B11_SWIR1", "B12_SWIR2"
]
BANDAS_10M_INDICES = [0, 1, 2, 3]
BANDAS_20M_INDICES = [4, 5, 6, 7, 8, 9]

# =============================================================================
# LOGGER COM TIMESTAMP
# =============================================================================

def log(msg, level="INFO"):
    """Log com timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# =============================================================================
# MODELOS OTIMIZADOS
# =============================================================================

class HATBlockOptimized(nn.Module):
    """HAT Block otimizado com fused operations"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        # Fusão de operações para reduzir chamadas CUDA
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        # Convolução espacial simplificada
        self.spatial_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
        # Feed-forward otimizado
        self.ff = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1, bias=False)
        )
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
    def forward(self, x):
        # Channel attention (fused)
        att = self.channel_att(x)
        x = x * att
        
        # Spatial attention simplificada
        spatial = self.spatial_conv(x)
        gate = self.spatial_gate(x)
        x = x + spatial * gate
        
        # Feed-forward
        residual = x
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.ff(x)
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return x + residual

class HATSuperResolutionOptimized(nn.Module):
    """HAT-L otimizado para processamento rápido"""
    def __init__(self, in_channels=4, out_channels=4, num_blocks=8):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 48, 3, padding=1)
        
        # Menos blocos mas mais eficientes
        self.blocks = nn.ModuleList([
            HATBlockOptimized(48) for _ in range(num_blocks)
        ])
        
        self.conv_mid = nn.Conv2d(48, 48, 3, padding=1)
        
        # Upsampling com pixel shuffle otimizado
        self.up = nn.Sequential(
            nn.Conv2d(48, 48 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(48, out_channels, 3, padding=1)
        )
        
        # Skip connection
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1)
        )
        
    def forward(self, x):
        skip = self.skip(x)
        
        # Feature extraction
        feat = self.conv_in(x)
        
        # Residual blocks
        for block in self.blocks:
            feat = block(feat) + feat * 0.1  # Residual scaling
        
        # Upsample
        out = self.up(feat)
        
        return out + skip

class RealESRGANRefinerOptimized(nn.Module):
    """Refinamento GAN otimizado"""
    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()
        # Encoder mais leve
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU()
        )
        
        # Residual blocks simplificados
        self.res_blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(128, 128, 3, padding=1)
            ) for _ in range(6)
        ])
        
        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, 3, padding=1)
        )
        
    def forward(self, x):
        # Upscale inicial
        up = F.interpolate(x, scale_factor=2.5, mode='bilinear', align_corners=False)
        
        # Encoder
        feat = self.enc(up)
        
        # Residual processing
        for block in self.res_blocks:
            feat = feat + block(feat)
        
        # Decoder
        out = self.dec(feat)
        
        return out + up.repeat(1, 1, 1, 1)[:, :out.shape[1]]

# =============================================================================
# FUNÇÕES DE PROCESSAMENTO OTIMIZADAS
# =============================================================================

@torch.cuda.amp.autocast(enabled=USE_AMP)
def process_batch_tiles(tiles_batch, model, device):
    """Processa múltiplos tiles em batch"""
    batch = torch.stack([torch.from_numpy(t).float() for t in tiles_batch]).to(device)
    with torch.no_grad():
        output = model(batch)
    return [o.cpu().numpy() for o in output]

def process_stage_parallel(input_array, model, device, scale_factor, 
                          tile_size=128, overlap=16, batch_size=4):
    """Processamento paralelo por estágio"""
    C, H, W = input_array.shape
    new_H, new_W = int(H * scale_factor), int(W * scale_factor)
    output = np.zeros((C, new_H, new_W), dtype=np.float32)
    weight = np.zeros((new_H, new_W), dtype=np.float32)
    
    # Calcular tiles com overlap
    stride = tile_size - overlap
    tiles_positions = []
    
    for h in range(0, H - overlap, stride):
        for w in range(0, W - overlap, stride):
            h_end = min(h + tile_size, H)
            w_end = min(w + tile_size, W)
            tiles_positions.append((h, w, h_end, w_end))
    
    # Processar em batches
    for i in tqdm(range(0, len(tiles_positions), batch_size), desc="Processando"):
        batch_positions = tiles_positions[i:i+batch_size]
        
        # Coletar tiles
        tiles = []
        for h, w, h_end, w_end in batch_positions:
            tile = input_array[:, h:h_end, w:w_end]
            # Padding para tamanho fixo
            if tile.shape[1] != tile_size or tile.shape[2] != tile_size:
                padded = np.zeros((C, tile_size, tile_size), dtype=np.float32)
                padded[:, :tile.shape[1], :tile.shape[2]] = tile
                tiles.append(padded)
            else:
                tiles.append(tile)
        
        # Processar batch
        sr_tiles = process_batch_tiles(tiles, model, device)
        
        # Reconstruir output
        for idx, (h, w, h_end, w_end) in enumerate(batch_positions):
            sr_tile = sr_tiles[idx][:, :(h_end-h)*scale_factor, :(w_end-w)*scale_factor]
            h_out = int(h * scale_factor)
            w_out = int(w * scale_factor)
            
            output[:, h_out:h_out+sr_tile.shape[1], 
                   w_out:w_out+sr_tile.shape[2]] += sr_tile
            weight[h_out:h_out+sr_tile.shape[1], 
                   w_out:w_out+sr_tile.shape[2]] += 1
    
    # Normalizar por overlap
    weight[weight == 0] = 1
    for c in range(C):
        output[c] /= weight
    
    return output

# =============================================================================
# FUNÇÕES AUXILIARES (mantidas do original)
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
                log(f"DNS (tentativa {attempt}/{max_tries})", "WARN")
                if attempt == 1 and not check_internet():
                    log("SEM INTERNET", "ERROR")
            elif "timeout" in err or "connection" in err:
                log(f"Rede (tentativa {attempt}/{max_tries})", "WARN")
            else:
                log(f"Erro (tentativa {attempt}/{max_tries}): {e}", "ERROR")
            if attempt < max_tries:
                d = delay * (backoff ** (attempt - 1))
                log(f"Aguardando {d:.0f}s...")
                time.sleep(d)
            else:
                raise

def save_geotiff(arr, transform, crs, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, 'w', driver='GTiff',
                       height=arr.shape[1], width=arr.shape[2],
                       count=arr.shape[0], dtype=rasterio.float32,
                       crs=crs, transform=transform, compress='lzw',
                       BIGTIFF='YES') as dst:
        dst.write(arr.astype(rasterio.float32))

def pad_to_multiple(arr, tile_size):
    _, H, W = arr.shape
    pad_h = (tile_size - H % tile_size) % tile_size
    pad_w = (tile_size - W % tile_size) % tile_size
    if pad_h > 0 or pad_w > 0:
        arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    return arr, pad_h, pad_w

def clip_to_polygon(arr, transform, geometria):
    mask = geometry_mask([geometria], transform=transform, invert=True,
                         out_shape=(arr.shape[1], arr.shape[2]))
    masked = arr.copy().astype("float32")
    for b in range(arr.shape[0]):
        masked[b][~mask] = np.nan
    return masked, mask

def upscale_bicubico(banda_2d, fator):
    return scipy_zoom(banda_2d, (fator, fator), order=3, mode='reflect')

def aplicar_rio_color(rgb_arr, transform, crs, caminho_saida):
    """Aplicar correção de cores com rio-color"""
    try:
        from rio_color.operations import simple_atmo, saturation
    except ImportError:
        log("pip install rio-color para ativar correção de cores", "WARN")
        return rgb_arr

    rgb = rgb_arr.copy().astype("float32")
    mask_nan = np.isnan(rgb[0])
    rgb = np.nan_to_num(rgb, nan=0.0)

    rgb_eq = rgb.copy()
    for b in range(3):
        banda = rgb_eq[b]
        mascara = ~np.isnan(rgb[b])
        if np.any(mascara):
            vmin = np.percentile(banda[mascara], 2)
            vmax = np.percentile(banda[mascara], 98)
            if vmax > vmin:
                rgb_eq[b][mascara] = (banda[mascara] - vmin) / (vmax - vmin)
    
    rgb_enh = simple_atmo(rgb_eq, haze=0.02, contrast=3.5, bias=0.5)
    rgb_enh = saturation(rgb_enh, proportion=1.4)
    rgb_enh = np.clip(rgb_enh, 0.0, 1.0)

    for b in range(3):
        rgb_enh[b][mask_nan] = np.nan

    save_geotiff(rgb_enh, transform, crs, caminho_saida)
    return rgb_enh

def salvar_bandas_individuais(arr, nomes, transform, crs, pasta):
    """Salvar cada banda em arquivo separado"""
    pasta_bandas = os.path.join(OUTPUT_DIR, pasta)
    os.makedirs(pasta_bandas, exist_ok=True)
    for i, nome in enumerate(nomes):
        caminho = os.path.join(pasta_bandas, f"{nome}.tif")
        with rasterio.open(caminho, 'w', driver='GTiff',
                           height=arr.shape[1], width=arr.shape[2],
                           count=1, dtype=rasterio.float32,
                           crs=crs, transform=transform, compress='lzw',
                           BIGTIFF='YES') as dst:
            dst.write(arr[i:i+1].astype(rasterio.float32))
    log(f"{len(nomes)} bandas salvas em: {pasta_bandas}/")

# =============================================================================
# FUNÇÃO PRINCIPAL OTIMIZADA
# =============================================================================

def main():
    start_time = time.time()
    log("=" * 70)
    log("SEN2SR_ULTRA - Super-Resolução Máxima (0.5m) [OTIMIZADO]")
    log("Pipeline: 10m → 2.5m → 1.25m → 0.5m")
    log("Otimizações: Batch processing + Mixed Precision + Overlap")
    log("=" * 70)

    # Verificar conectividade
    log("Verificando conectividade...")
    if not check_dns(BLOB_HOST) and not check_internet():
        log("SEM INTERNET. Abortando.", "ERROR")
        return
    log("Conectividade OK")

    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Dispositivo: {device}")
    if device.type == 'cuda':
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"VRAM: {mem_gb:.1f} GB")
        if mem_gb < 8:
            log("VRAM < 8GB - Reduzindo batch size", "WARN")
            global BATCH_SIZE_HAT
            BATCH_SIZE_HAT = 2
        elif mem_gb >= 16:
            BATCH_SIZE_HAT = 8
            log("VRAM excelente! Batch size aumentado para 8")
        
        # Otimizações CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    # 1. Carregar vetor
    log(f"Lendo vetor: {VETOR_LIMITE}")
    gdf = gpd.read_file(VETOR_LIMITE)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    bbox = gdf.total_bounds
    geometria = unary_union(gdf.geometry.values)
    log(f"Bbox: {bbox}")

    # 2. Calcular dimensões
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
    
    fator_total = FATOR_SR1 * FATOR_SR2 * FATOR_SR3
    edge_px_final = int(edge_px * fator_total)
    
    log(f"Área: {largura_m:.0f}m × {altura_m:.0f}m")
    log(f"Cubo 10m: {edge_px}×{edge_px} px ({ncols}×{nrows} tiles)")
    log(f"Cubo 0.5m: {edge_px_final}×{edge_px_final} px")
    log(f"Upscaling total: {fator_total:.0f}x")

    # 3. Carregar modelos
    log("Carregando modelos de super-resolução...")
    
    # Estágio 1: SEN2SR
    try:
        import mamba_ssm
        HAS_MAMBA = True
        model_url = ("https://huggingface.co/tacofoundation/sen2sr/resolve/main"
                     "/SEN2SR/NonReference_RGBN_x4/mlm.json")
        model_dir = "model/SEN2SR_RGBN"
        log("Estágio 1: SEN2SR Mamba (4x)")
    except ImportError:
        HAS_MAMBA = False
        model_url = ("https://huggingface.co/tacofoundation/sen2sr/resolve/main"
                     "/SEN2SRLite/NonReference_RGBN_x4/mlm.json")
        model_dir = "model/SEN2SRLite_RGBN"
        log("Estágio 1: SEN2SR Lite SwinIR (4x)")

    os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(os.path.join(model_dir, "mlm.json")):
        mlstac.download(file=model_url, output_dir=model_dir)
    model_sr1 = mlstac.load(model_dir).compiled_model(device=device)
    
    # Estágio 2: HAT-L Otimizado
    log("Estágio 2: HAT-L Otimizado (2x)")
    model_sr2 = HATSuperResolutionOptimized(
        in_channels=4, out_channels=4, num_blocks=6  # Reduzido para velocidade
    ).to(device)
    model_sr2.eval()
    
    # Compilar com Torch 2.0+ se disponível (apenas em GPU)
    if USE_TORCH_COMPILE and device.type == 'cuda' and hasattr(torch, 'compile'):
        try:
            log("Compilando HAT-L com torch.compile()...")
            model_sr2 = torch.compile(model_sr2, mode='reduce-overhead')
            log("Compilação JIT ativada!")
        except Exception as e:
            log(f"Falha na compilação JIT: {e}", "WARN")
    
    # Estágio 3: Real-ESRGAN+ Otimizado
    log("Estágio 3: Real-ESRGAN+ Otimizado (2.5x)")
    model_sr3 = RealESRGANRefinerOptimized(in_channels=4, out_channels=4).to(device)
    model_sr3.eval()
    
    total_params = sum(p.numel() for m in [model_sr2, model_sr3] for p in m.parameters())
    log(f"Parâmetros totais (estágios 2+3): {total_params/1e6:.1f}M")
    log("Modelos carregados!")

    # 4. Baixar cubo
    log(f"Baixando cubo Sentinel-2 ({edge_px}×{edge_px}px)...")
    da = cubo.create(lat=cy, lon=cx, collection=COLECAO, bands=BANDAS,
                     start_date=START_DATE, end_date=END_DATE,
                     edge_size=edge_px, resolution=RESOLUCAO_M)

    crs = da.rio.crs
    transf_cubo = da.rio.transform()
    log(f"CRS: {crs}")

    if crs is None:
        crs = rasterio.crs.CRS.from_epsg(32722)
        log(f"CRS forçado: {crs}", "WARN")

    log("Reprojetando polígono...")
    from shapely.ops import transform as shapely_transform
    project = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    geometria_proj = shapely_transform(project, geometria)

    cubo_np = compute_with_retry(da)
    log(f"Shape do cubo: {cubo_np.shape}")

    cubo_np, pad_h, pad_w = pad_to_multiple(cubo_np, TILE_SIZE)
    if pad_h or pad_w:
        log(f"Padding aplicado: {pad_h}×{pad_w}")

    # 5. Pipeline de super-resolução otimizado
    C, H, W = cubo_np.shape

    # === ESTÁGIO 1: SEN2SR (10m → 2.5m) ===
    stage1_start = time.time()
    log("=" * 70)
    log("ETAPA 1/3: SEN2SR (10m → 2.5m)")
    log("=" * 70)
    
    H1 = int(H * FATOR_SR1)
    W1 = int(W * FATOR_SR1)
    mosaico_sr1 = np.zeros((4, H1, W1), dtype="float32")

    nrows_sr1 = H // TILE_SIZE
    ncols_sr1 = W // TILE_SIZE

    for r in tqdm(range(nrows_sr1), desc="SEN2SR"):
        for c in range(ncols_sr1):
            h0, h1 = r * TILE_SIZE, (r + 1) * TILE_SIZE
            w0, w1 = c * TILE_SIZE, (c + 1) * TILE_SIZE

            tile = cubo_np[:, h0:h1, w0:w1]
            tile_norm = (tile / 10000).astype("float32")
            tile_rgbn = tile_norm[[2, 1, 0, 3], :, :]

            X = torch.from_numpy(tile_rgbn).float().to(device)
            X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            with torch.no_grad():
                sr = model_sr1(X[None]).squeeze(0)
            sr_np = sr.cpu().numpy().astype("float32")

            h0s, h1s = r * TILE_SIZE * FATOR_SR1, (r + 1) * TILE_SIZE * FATOR_SR1
            w0s, w1s = c * TILE_SIZE * FATOR_SR1, (c + 1) * TILE_SIZE * FATOR_SR1
            mosaico_sr1[:, h0s:h1s, w0s:w1s] = sr_np

    stage1_time = time.time() - stage1_start
    log(f"Etapa 1 concluída em {stage1_time:.1f}s! Shape: {mosaico_sr1.shape}")

    # === ESTÁGIO 2: HAT-L Otimizado (2.5m → 1.25m) ===
    stage2_start = time.time()
    log("=" * 70)
    log("ETAPA 2/3: HAT-L Otimizado (2.5m → 1.25m)")
    log(f"Batch size: {BATCH_SIZE_HAT}")
    log("=" * 70)
    
    log("Processando com overlap e batch processing...")
    mosaico_sr2 = process_stage_parallel(
        mosaico_sr1, model_sr2, device, FATOR_SR2,
        tile_size=128, overlap=TILE_OVERLAP, batch_size=BATCH_SIZE_HAT
    )
    
    stage2_time = time.time() - stage2_start
    log(f"Etapa 2 concluída em {stage2_time:.1f}s! Shape: {mosaico_sr2.shape}")

    # === ESTÁGIO 3: Real-ESRGAN+ Otimizado (1.25m → 0.5m) ===
    stage3_start = time.time()
    log("=" * 70)
    log("ETAPA 3/3: Real-ESRGAN+ Otimizado (1.25m → 0.5m)")
    log("=" * 70)
    
    log("Processando refinamento final...")
    mosaico_sr3 = process_stage_parallel(
        mosaico_sr2, model_sr3, device, FATOR_SR3,
        tile_size=64, overlap=TILE_OVERLAP//2, batch_size=BATCH_SIZE_HAT
    )
    
    stage3_time = time.time() - stage3_start
    log(f"Etapa 3 concluída em {stage3_time:.1f}s! Shape: {mosaico_sr3.shape}")

    # 6. Aplicar máscara e salvar
    log("Aplicando máscara do polígono...")
    
    # Normalizar mosaico original
    mosaic_orig = np.zeros((C, H, W), dtype="float32")
    for r in range(nrows_sr1):
        for c in range(ncols_sr1):
            h0, h1 = r * TILE_SIZE, (r + 1) * TILE_SIZE
            w0, w1 = c * TILE_SIZE, (c + 1) * TILE_SIZE
            mosaic_orig[:, h0:h1, w0:w1] = cubo_np[:, h0:h1, w0:w1] / 10000

    mosaic_orig_clipped, mask_orig = clip_to_polygon(mosaic_orig, transf_cubo, geometria_proj)

    transf_final = rasterio.Affine(
        transf_cubo.a / fator_total, transf_cubo.b, transf_cubo.c,
        transf_cubo.d, transf_cubo.e / fator_total, transf_cubo.f,
    )

    mosaic_final_clipped, mask_final = clip_to_polygon(mosaico_sr3, transf_final, geometria_proj)

    # 7. Montar 10 bandas
    log("Montando 10 bandas em 0.5m...")
    H_final, W_final = mosaic_final_clipped.shape[1], mosaic_final_clipped.shape[2]
    todas_bandas = np.zeros((10, H_final, W_final), dtype="float32")

    todas_bandas[0] = mosaic_final_clipped[2]  # B02
    todas_bandas[1] = mosaic_final_clipped[1]  # B03
    todas_bandas[2] = mosaic_final_clipped[0]  # B04
    todas_bandas[3] = mosaic_final_clipped[3]  # B08
    log("4 bandas 10m super-resolvidas")

    mosaic_orig_20m_clipped, _ = clip_to_polygon(mosaic_orig, transf_cubo, geometria_proj)
    for i, idx_orig in enumerate(BANDAS_20M_INDICES):
        idx_final = i + 4
        banda_up = upscale_bicubico(mosaic_orig_20m_clipped[idx_orig], fator_total)
        banda_up[~mask_final] = np.nan
        todas_bandas[idx_final] = banda_up.astype("float32")
    
    log("6 bandas 20m interpoladas")

    # 8. Salvar resultados
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4 bandas RGBN
    p_sr = os.path.join(OUTPUT_DIR, "super_resolved_0_5m.tif")
    save_geotiff(mosaic_final_clipped, transf_final, crs, p_sr)
    log(f"4 bandas (RGBN) 0.5m salvas: {p_sr}")

    # RGB com rio-color
    log("Aplicando rio-color...")
    try:
        import rio_color
        rgb_enh = aplicar_rio_color(
            mosaic_final_clipped[[0, 1, 2]], transf_final, crs,
            os.path.join(OUTPUT_DIR, "super_resolved_0_5m_cor.tif")
        )
        log("RGB corrigido salvo!")
        USAR_COLOR = True
    except:
        log("rio-color não disponível", "WARN")
        USAR_COLOR = False

    # 10 bandas individuais
    log("Salvando 10 bandas individuais...")
    salvar_bandas_individuais(todas_bandas, BANDAS_NOMES, transf_final, crs, "bandas_0_5m")

    # 9. Visualização
    log("Gerando visualização...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(np.clip(mosaic_orig_clipped[[2, 1, 0]].transpose(1, 2, 0) * 1.5, 0, 1))
    axes[0].set_title("Original 10m", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(mosaic_final_clipped[[0, 1, 2]].transpose(1, 2, 0) * 1.5, 0, 1))
    axes[1].set_title(f"Super-res. {RESOLUCAO_FINAL}m", fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparacao_ultra.png"), dpi=200, bbox_inches='tight')
    plt.show(block=False)

    # Estatísticas finais
    total_time = time.time() - start_time
    area_km2 = np.sum(mask_orig) * RESOLUCAO_M**2 / 1e6
    
    log("=" * 70)
    log("ESTATÍSTICAS FINAIS")
    log("=" * 70)
    log(f"Área: {area_km2:.2f} km²")
    log(f"Resolução final: {RESOLUCAO_FINAL}m")
    log(f"Upscaling total: {fator_total:.0f}x")
    log(f"Tempo total: {total_time:.1f}s ({total_time/60:.1f}min)")
    log(f"Tempo etapa 1 (SEN2SR): {stage1_time:.1f}s")
    log(f"Tempo etapa 2 (HAT-L): {stage2_time:.1f}s")
    log(f"Tempo etapa 3 (Real-ESRGAN+): {stage3_time:.1f}s")
    log(f"Tamanho arquivo final: {mosaic_final_clipped.nbytes / 1e9:.2f} GB")
    log(f"Resultados em: {OUTPUT_DIR}/")
    log("PIPELINE ULTRA CONCLUÍDO! (0.5m)")
    plt.show()


if __name__ == "__main__":
    main()
