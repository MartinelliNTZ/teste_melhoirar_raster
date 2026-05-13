# -*- coding: utf-8 -*-
"""
SEN2SR_ULTRA_V3 - Super-resolução REALISTA com modelos PRÉ-TREINADOS
Pipeline: 10m → 2.5m (SEN2SR) → 1.25m (Swin2SR x2)
+ Fusão multi-temporal para melhoria de qualidade
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
import warnings
from datetime import datetime, timedelta
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
MAX_IMAGES   = 3             # Número de imagens para fusão temporal

COLECAO      = "sentinel-2-l2a"
RESOLUCAO_M  = 10
TILE_SIZE    = 128
OUTPUT_DIR   = "resultados_ultra_v3"
BLOB_HOST    = "sentinel2l2a01.blob.core.windows.net"

# Pipeline REALISTA com modelos PRÉ-TREINADOS
FATOR_SR1    = 4        # 10m → 2.5m (SEN2SR - TREINADO para Sentinel-2)
FATOR_SR2    = 2        # 2.5m → 1.25m (Swin2SR x2 - TREINADO para imagens reais)
RESOLUCAO_FINAL = 1.25  # Resolução alvo REALISTA em metros

# Otimizações
BATCH_SIZE = 4
TILE_OVERLAP = 16

BANDAS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]
BANDAS_NOMES = [
    "B02_Blue", "B03_Green", "B04_Red", "B08_NIR",
    "B05_RedEdge1", "B06_RedEdge2", "B07_RedEdge3",
    "B8A_NarrowNIR", "B11_SWIR1", "B12_SWIR2"
]
BANDAS_10M_INDICES = [0, 1, 2, 3]
BANDAS_20M_INDICES = [4, 5, 6, 7, 8, 9]

# =============================================================================
# LOGGER
# =============================================================================

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "ℹ️", "WARN": "⚠️", "ERROR": "❌", "SUCCESS": "✅"}.get(level, "")
    print(f"[{timestamp}] {prefix} {msg}")

# =============================================================================
# MODELOS PRÉ-TREINADOS
# =============================================================================

class Swin2SRx2:
    """Wrapper para Swin2SR 2x pré-treinado"""
    def __init__(self, device):
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Carrega Swin2SR 2x do HuggingFace"""
        try:
            from transformers import Swin2SRForImageSuperResolution
            from transformers import Swin2SRImageProcessor
            
            log("Carregando Swin2SR 2x pré-treinado (Real-World SR)...")
            
            # Modelo 2x - mais conservador, menos alucinações
            model_id = "caidas/swin2SR-realworld-sr-x2-64-bsrgan-psnr"
            
            self.processor = Swin2SRImageProcessor.from_pretrained(model_id)
            self.model = Swin2SRForImageSuperResolution.from_pretrained(
                model_id,
                torch_dtype=torch.float32
            ).to(self.device)
            self.model.eval()
            
            params = sum(p.numel() for p in self.model.parameters()) / 1e6
            log(f"✅ Swin2SR 2x carregado! Parâmetros: {params:.1f}M")
            log(f"   Upscaling: 2x (2.5m → 1.25m)")
            
        except ImportError:
            log("Transformers não instalado.", "ERROR")
            log("Execute: pip install transformers timm", "ERROR")
            raise
        except Exception as e:
            log(f"Erro ao carregar Swin2SR 2x: {e}", "ERROR")
            log("Tentando modelo alternativo...", "WARN")
            self._load_fallback()
    
    def _load_fallback(self):
        """Fallback: usa modelo 4x com downscale ou bicubic"""
        log("Usando bicubic de alta qualidade como fallback...")
        self.model = None
        self.processor = None
    
    def __call__(self, x):
        """Processa tensor ou numpy array"""
        if self.model is None:
            # Fallback: bicubic de alta qualidade
            if isinstance(x, np.ndarray):
                result = np.zeros((x.shape[0], x.shape[1]*2, x.shape[2]*2) if x.ndim == 3 
                                  else (x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3]*2),
                                  dtype=np.float32)
                if x.ndim == 3:
                    for c in range(x.shape[0]):
                        result[c] = scipy_zoom(x[c], 2.0, order=3)
                else:
                    for b in range(x.shape[0]):
                        for c in range(x.shape[1]):
                            result[b, c] = scipy_zoom(x[b, c], 2.0, order=3)
                return torch.from_numpy(result).to(self.device)
            else:
                return F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            outputs = []
            for i in range(x.shape[0]):
                img = x[i].cpu().numpy().transpose(1, 2, 0)
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                
                inputs = self.processor(img, return_tensors="pt").to(self.device)
                output = self.model(**inputs)
                output = output.reconstruction.squeeze(0).cpu().numpy()
                output = output.transpose(2, 0, 1) / 255.0
                outputs.append(torch.from_numpy(output))
            
            return torch.stack(outputs).to(self.device)
    
    def to(self, device):
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self

# =============================================================================
# PROCESSAMENTO MULTI-TEMPORAL
# =============================================================================

class MultiTemporalFusion:
    """Fusão de múltiplas imagens temporais"""
    
    @staticmethod
    def compute_quality_score(image):
        """Score de nitidez local"""
        grad_x = np.abs(np.diff(image, axis=2))
        grad_y = np.abs(np.diff(image, axis=1))
        
        grad_x = np.pad(grad_x, ((0, 0), (0, 0), (0, 1)), mode='edge')
        grad_y = np.pad(grad_y, ((0, 0), (0, 1), (0, 0)), mode='edge')
        
        score = np.sqrt(grad_x**2 + grad_y**2).mean(axis=0)
        return score
    
    @staticmethod
    def fuse_images(images, method='weighted_median'):
        """Fusiona múltiplas imagens"""
        if len(images) == 1:
            return images[0]
        
        images_stack = np.stack(images, axis=0)
        
        if method == 'weighted_median':
            weights = []
            for img in images:
                score = MultiTemporalFusion.compute_quality_score(img)
                weights.append(score)
            
            weights = np.stack(weights, axis=0)
            weights = weights / (weights.sum(axis=0, keepdims=True) + 1e-8)
            
            fused = np.sum(images_stack * weights[:, np.newaxis, :, :], axis=0)
            return fused
        
        elif method == 'median':
            return np.nanmedian(images_stack, axis=0)
        
        else:
            return np.nanmean(images_stack, axis=0)

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def apply_blend_window(tile_size, overlap):
    """Janela de blending"""
    window = np.ones((tile_size, tile_size), dtype=np.float32)
    ramp = np.linspace(0, 1, overlap)
    
    window[:overlap, :] *= ramp[:, np.newaxis]
    window[-overlap:, :] *= ramp[::-1, np.newaxis]
    window[:, :overlap] *= ramp[np.newaxis, :]
    window[:, -overlap:] *= ramp[np.newaxis, ::-1]
    
    return window

def diagnosticar_tensor(arr, nome):
    """Diagnóstico de valores"""
    log(f"📊 {nome}:")
    log(f"   Shape: {arr.shape} | Min: {np.nanmin(arr):.4f} | Max: {np.nanmax(arr):.4f}")
    log(f"   Mean: {np.nanmean(arr):.4f} | NaN: {np.isnan(arr).sum()} | Inf: {np.isinf(arr).sum()}")

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

def compute_with_retry(da, idx=0, max_tries=5, delay=2.0, backoff=2.0):
    for attempt in range(1, max_tries + 1):
        try:
            return da[idx].compute().to_numpy()
        except Exception as e:
            err = str(e).lower()
            if "resolve" in err or "timeout" in err or "connection" in err:
                log(f"Rede (tentativa {attempt}/{max_tries})", "WARN")
            else:
                log(f"Erro (tentativa {attempt}/{max_tries}): {e}", "ERROR")
            if attempt < max_tries:
                d = delay * (backoff ** (attempt - 1))
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

def process_tile_sr(tile, model, device):
    """Processa um tile com o modelo SR"""
    X = torch.from_numpy(tile).float().to(device)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    with torch.no_grad():
        if X.dim() == 3:
            X = X.unsqueeze(0)
        sr = model(X).squeeze(0)
    
    return sr.cpu().numpy().astype("float32")

def baixar_cubos_temporais(cy, cx, edge_px, start_date, end_date, max_images=3):
    """Baixa múltiplas imagens em diferentes períodos"""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days
    
    if total_days < 10:
        log("Janela muito curta, baixando imagem única...", "WARN")
        return baixar_cubo_unico(cy, cx, edge_px, start_date, end_date)
    
    interval = max(total_days // max_images, 10)
    
    cubos_temporais = []
    crs = None
    transf_cubo = None
    
    for i in range(max_images):
        center_date = start_dt + timedelta(days=i * interval + interval // 2)
        if center_date > end_dt:
            center_date = end_dt
        
        sub_start = (center_date - timedelta(days=3)).strftime("%Y-%m-%d")
        sub_end = (center_date + timedelta(days=3)).strftime("%Y-%m-%d")
        
        log(f"Buscando imagem {i+1}/{max_images}: {sub_start} a {sub_end}")
        
        try:
            da = cubo.create(
                lat=cy, lon=cx,
                collection=COLECAO,
                bands=BANDAS,
                start_date=sub_start,
                end_date=sub_end,
                edge_size=edge_px,
                resolution=RESOLUCAO_M
            )
            
            if crs is None:
                crs = da.rio.crs
                transf_cubo = da.rio.transform()
                if crs is None:
                    crs = rasterio.crs.CRS.from_epsg(32722)
            
            cubo_np = compute_with_retry(da, idx=0, max_tries=3)
            cubo_np, pad_h, pad_w = pad_to_multiple(cubo_np, TILE_SIZE)
            
            cubo_norm = cubo_np / 10000.0
            cubos_temporais.append(cubo_norm)
            
            log(f"   ✅ Shape: {cubo_np.shape}")
            
        except Exception as e:
            log(f"   ⚠️ Falha: {e}", "WARN")
            continue
    
    return cubos_temporais, crs, transf_cubo

def baixar_cubo_unico(cy, cx, edge_px, start_date, end_date):
    """Fallback: baixa uma única imagem"""
    log(f"Baixando: {start_date} a {end_date}")
    
    da = cubo.create(
        lat=cy, lon=cx,
        collection=COLECAO,
        bands=BANDAS,
        start_date=start_date,
        end_date=end_date,
        edge_size=edge_px,
        resolution=RESOLUCAO_M
    )
    
    crs = da.rio.crs
    transf_cubo = da.rio.transform()
    if crs is None:
        crs = rasterio.crs.CRS.from_epsg(32722)
    
    cubo_np = compute_with_retry(da, idx=0, max_tries=3)
    cubo_np, pad_h, pad_w = pad_to_multiple(cubo_np, TILE_SIZE)
    
    cubo_norm = cubo_np / 10000.0
    log(f"✅ Shape: {cubo_np.shape}")
    
    return [cubo_norm], crs, transf_cubo

# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def main():
    start_time = time.time()
    log("=" * 70)
    log("SEN2SR_ULTRA_V3 - Super-Resolução REALISTA")
    log("Pipeline: 10m → 2.5m (SEN2SR) → 1.25m (Swin2SR 2x)")
    log("Modelos: 100% PRÉ-TREINADOS | Resolução final: 1.25m")
    log("=" * 70)

    # Verificar conectividade
    if not check_dns(BLOB_HOST) and not check_internet():
        log("SEM INTERNET. Abortando.", "ERROR")
        return
    log("Conectividade OK")

    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Dispositivo: {device}")
    if device.type == 'cuda':
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
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
    
    fator_total = FATOR_SR1 * FATOR_SR2  # 4 * 2 = 8
    edge_px_final = int(edge_px * fator_total)
    
    log(f"Área: {largura_m:.0f}m × {altura_m:.0f}m")
    log(f"Cubo 10m: {edge_px}×{edge_px} px ({ncols}×{nrows} tiles)")
    log(f"Cubo 1.25m: {edge_px_final}×{edge_px_final} px")
    log(f"Upscaling total: {fator_total}x (8x)")

    # 3. Carregar modelos PRÉ-TREINADOS
    log("=" * 70)
    log("CARREGANDO MODELOS PRÉ-TREINADOS")
    log("=" * 70)
    
    # Estágio 1: SEN2SR (TREINADO para Sentinel-2)
    try:
        import mamba_ssm
        model_url = ("https://huggingface.co/tacofoundation/sen2sr/resolve/main"
                     "/SEN2SR/NonReference_RGBN_x4/mlm.json")
        model_dir = "model/SEN2SR_RGBN"
        log("✅ SEN2SR Mamba (4x) - TREINADO para Sentinel-2")
    except ImportError:
        model_url = ("https://huggingface.co/tacofoundation/sen2sr/resolve/main"
                     "/SEN2SRLite/NonReference_RGBN_x4/mlm.json")
        model_dir = "model/SEN2SRLite_RGBN"
        log("✅ SEN2SR Lite SwinIR (4x) - TREINADO para Sentinel-2")

    os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(os.path.join(model_dir, "mlm.json")):
        mlstac.download(file=model_url, output_dir=model_dir)
    model_sr1 = mlstac.load(model_dir).compiled_model(device=device)
    
    # Estágio 2: Swin2SR 2x (TREINADO para imagens reais)
    log("Carregando Swin2SR 2x...")
    model_sr2 = Swin2SRx2(device)
    log("✅ Swin2SR 2x (2x) - TREINADO para imagens reais")
    
    log("=" * 70)
    log("📊 RESUMO DOS MODELOS:")
    log("   Estágio 1: SEN2SR - Treinado ESPECIFICAMENTE para Sentinel-2")
    log("   Estágio 2: Swin2SR 2x - Treinado para imagens reais (conservador)")
    log("   Upscaling: 4x → 2x = 8x total")
    log("   Resolução: 10m → 2.5m → 1.25m")
    log("=" * 70)

    # 4. Baixar imagens
    log("Baixando imagens Sentinel-2...")
    
    cubos_temporais, crs, transf_cubo = baixar_cubos_temporais(
        cy, cx, edge_px, START_DATE, END_DATE, MAX_IMAGES
    )
    
    if len(cubos_temporais) == 0:
        log("Nenhuma imagem disponível! Abortando.", "ERROR")
        return
    
    log(f"✅ {len(cubos_temporais)} imagens baixadas")
    
    # Reprojetar polígono
    from shapely.ops import transform as shapely_transform
    project = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    geometria_proj = shapely_transform(project, geometria)
    
    C, H, W = cubos_temporais[0].shape

    # 5. Processar cada imagem
    log("=" * 70)
    log("PROCESSANDO IMAGENS")
    log("=" * 70)
    
    resultados_temporais = []
    
    for img_idx, cubo_norm in enumerate(cubos_temporais):
        log(f"\n{'='*50}")
        log(f"Imagem {img_idx+1}/{len(cubos_temporais)}")
        log(f"{'='*50}")
        
        # === ESTÁGIO 1: SEN2SR (10m → 2.5m) ===
        etapa1_start = time.time()
        log("Etapa 1/2: SEN2SR (10m → 2.5m)")
        
        H1 = int(H * FATOR_SR1)
        W1 = int(W * FATOR_SR1)
        mosaico_sr1 = np.zeros((4, H1, W1), dtype="float32")
        
        for r in tqdm(range(nrows), desc="SEN2SR"):
            for c in range(ncols):
                h0, h1 = r * TILE_SIZE, (r + 1) * TILE_SIZE
                w0, w1 = c * TILE_SIZE, (c + 1) * TILE_SIZE
                
                tile = cubo_norm[:, h0:h1, w0:w1]
                tile_rgbn = tile[[2, 1, 0, 3], :, :]  # RGBN order
                
                sr_np = process_tile_sr(tile_rgbn, model_sr1, device)
                
                h0s, h1s = r * TILE_SIZE * FATOR_SR1, (r + 1) * TILE_SIZE * FATOR_SR1
                w0s, w1s = c * TILE_SIZE * FATOR_SR1, (c + 1) * TILE_SIZE * FATOR_SR1
                mosaico_sr1[:, h0s:h1s, w0s:w1s] = sr_np
        
        mosaico_sr1 = np.clip(mosaico_sr1, 0.0, 1.0)
        etapa1_time = time.time() - etapa1_start
        diagnosticar_tensor(mosaico_sr1, f"Imagem {img_idx+1} - SEN2SR (2.5m)")
        log(f"✅ Etapa 1 concluída em {etapa1_time:.1f}s")
        
        # Salvar etapa 1
        transf_sr1 = rasterio.Affine(
            transf_cubo.a / FATOR_SR1, transf_cubo.b, transf_cubo.c,
            transf_cubo.d, transf_cubo.e / FATOR_SR1, transf_cubo.f,
        )
        p_sr1 = os.path.join(OUTPUT_DIR, f"etapa1_sen2sr_2_5m_img{img_idx+1}.tif")
        save_geotiff(mosaico_sr1, transf_sr1, crs, p_sr1)
        
        # === ESTÁGIO 2: Swin2SR 2x (2.5m → 1.25m) ===
        etapa2_start = time.time()
        log("Etapa 2/2: Swin2SR 2x (2.5m → 1.25m)")
        
        H2 = int(H1 * FATOR_SR2)
        W2 = int(W1 * FATOR_SR2)
        mosaico_sr2 = np.zeros((4, H2, W2), dtype="float32")
        weight = np.zeros((H2, W2), dtype=np.float32)
        
        tile_size_sr2 = 64
        overlap_sr2 = 8
        blend_window = apply_blend_window(tile_size_sr2, overlap_sr2)
        stride = tile_size_sr2 - overlap_sr2
        
        tiles_list = []
        for h in range(0, H1 - overlap_sr2, stride):
            for w in range(0, W1 - overlap_sr2, stride):
                h_end = min(h + tile_size_sr2, H1)
                w_end = min(w + tile_size_sr2, W1)
                tiles_list.append((h, w, h_end, w_end))
        
        for i in tqdm(range(0, len(tiles_list), BATCH_SIZE), desc="Swin2SR 2x"):
            batch_positions = tiles_list[i:i+BATCH_SIZE]
            
            batch_tiles = []
            for h, w, h_end, w_end in batch_positions:
                tile = mosaico_sr1[:, h:h_end, w:w_end]
                if tile.shape[1] != tile_size_sr2 or tile.shape[2] != tile_size_sr2:
                    padded = np.zeros((4, tile_size_sr2, tile_size_sr2), dtype=np.float32)
                    padded[:, :tile.shape[1], :tile.shape[2]] = tile
                    batch_tiles.append(padded)
                else:
                    batch_tiles.append(tile)
            
            batch_tensor = torch.stack([torch.from_numpy(t).float() for t in batch_tiles]).to(device)
            with torch.no_grad():
                sr_tiles = model_sr2(batch_tensor).cpu().numpy()
            
            for idx, (h, w, h_end, w_end) in enumerate(batch_positions):
                sr_tile = sr_tiles[idx]
                tile_h = int((h_end - h) * FATOR_SR2)
                tile_w = int((w_end - w) * FATOR_SR2)
                sr_tile = sr_tile[:, :tile_h, :tile_w]
                
                h_out = int(h * FATOR_SR2)
                w_out = int(w * FATOR_SR2)
                
                window_resized = scipy_zoom(blend_window, (tile_h/tile_size_sr2, tile_w/tile_size_sr2), order=1)
                
                for ch in range(4):
                    mosaico_sr2[ch, h_out:h_out+tile_h, w_out:w_out+tile_w] += sr_tile[ch] * window_resized
                weight[h_out:h_out+tile_h, w_out:w_out+tile_w] += window_resized
        
        weight[weight == 0] = 1
        for ch in range(4):
            mosaico_sr2[ch] /= weight
        
        mosaico_sr2 = np.clip(mosaico_sr2, 0.0, 1.0)
        etapa2_time = time.time() - etapa2_start
        diagnosticar_tensor(mosaico_sr2, f"Imagem {img_idx+1} - Swin2SR (1.25m)")
        log(f"✅ Etapa 2 concluída em {etapa2_time:.1f}s")
        
        resultados_temporais.append(mosaico_sr2)
    
    # 6. Fusão temporal
    log("=" * 70)
    log("FUSÃO TEMPORAL")
    log("=" * 70)
    
    if len(resultados_temporais) > 1:
        log(f"Fusionando {len(resultados_temporais)} imagens (weighted median)...")
        mosaico_final = MultiTemporalFusion.fuse_images(
            resultados_temporais, 
            method='weighted_median'
        )
        log("✅ Fusão temporal concluída!")
    else:
        mosaico_final = resultados_temporais[0]
        log("Apenas 1 imagem disponível, sem fusão")
    
    # 7. Aplicar máscara e salvar
    log("Aplicando máscara do polígono...")
    
    transf_final = rasterio.Affine(
        transf_cubo.a / fator_total, transf_cubo.b, transf_cubo.c,
        transf_cubo.d, transf_cubo.e / fator_total, transf_cubo.f,
    )
    
    mosaic_final_clipped, mask_final = clip_to_polygon(
        mosaico_final, transf_final, geometria_proj
    )
    
    # Original para referência
    mosaic_orig = cubos_temporais[0][[2, 1, 0, 3], :, :]
    mosaic_orig_clipped, mask_orig = clip_to_polygon(
        mosaic_orig, transf_cubo, geometria_proj
    )
    
    # 8. Salvar resultados
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Resultado principal
    p_sr = os.path.join(OUTPUT_DIR, f"super_resolved_{RESOLUCAO_FINAL}m.tif")
    save_geotiff(mosaic_final_clipped, transf_final, crs, p_sr)
    log(f"✅ Resultado salvo: {p_sr}")
    
    # 9. Montar 10 bandas
    log("Montando 10 bandas em 1.25m...")
    H_final, W_final = mosaic_final_clipped.shape[1], mosaic_final_clipped.shape[2]
    todas_bandas = np.zeros((10, H_final, W_final), dtype="float32")
    
    # 4 bandas super-resolvidas
    todas_bandas[0] = mosaic_final_clipped[2]  # B02 Blue
    todas_bandas[1] = mosaic_final_clipped[1]  # B03 Green
    todas_bandas[2] = mosaic_final_clipped[0]  # B04 Red
    todas_bandas[3] = mosaic_final_clipped[3]  # B08 NIR
    
    # 6 bandas 20m interpoladas
    mosaic_orig_20m_clipped, _ = clip_to_polygon(mosaic_orig, transf_cubo, geometria_proj)
    for i, idx_orig in enumerate(BANDAS_20M_INDICES):
        idx_final = i + 4
        banda_up = upscale_bicubico(mosaic_orig_20m_clipped[idx_orig], fator_total)
        banda_up[~mask_final] = np.nan
        todas_bandas[idx_final] = banda_up.astype("float32")
    
    log("10 bandas montadas")
    
    # Salvar bandas individuais
    pasta_bandas = os.path.join(OUTPUT_DIR, "bandas_1_25m")
    os.makedirs(pasta_bandas, exist_ok=True)
    for i, nome in enumerate(BANDAS_NOMES):
        caminho = os.path.join(pasta_bandas, f"{nome}.tif")
        save_geotiff(todas_bandas[i:i+1], transf_final, crs, caminho)
    log(f"✅ 10 bandas salvas em: {pasta_bandas}/")
    
    # 10. Visualização
    log("Gerando visualização...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Original 10m
    axes[0, 0].imshow(np.clip(mosaic_orig_clipped[[0, 1, 2]].transpose(1, 2, 0) * 2.5, 0, 1))
    axes[0, 0].set_title("Original 10m", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # SEN2SR 2.5m
    sr1_display = resultados_temporais[0] if len(resultados_temporais) > 0 else mosaico_sr1
    axes[0, 1].imshow(np.clip(sr1_display[[0, 1, 2]].transpose(1, 2, 0) * 2.0, 0, 1))
    axes[0, 1].set_title("SEN2SR (2.5m)", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Swin2SR 1.25m (bruto)
    axes[1, 0].imshow(np.clip(mosaico_final[[0, 1, 2]].transpose(1, 2, 0) * 2.0, 0, 1))
    axes[1, 0].set_title(f"Swin2SR 2x ({RESOLUCAO_FINAL}m)", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Final com máscara
    axes[1, 1].imshow(np.clip(mosaic_final_clipped[[0, 1, 2]].transpose(1, 2, 0) * 2.0, 0, 1))
    axes[1, 1].set_title(f"Final com máscara ({RESOLUCAO_FINAL}m)", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "00_COMPARACAO_FINAL.png"), dpi=200, bbox_inches='tight')
    log("✅ Visualização salva")
    
    # 11. Estatísticas
    total_time = time.time() - start_time
    area_km2 = np.sum(mask_orig) * RESOLUCAO_M**2 / 1e6
    
    log("=" * 70)
    log("✅ PIPELINE CONCLUÍDO!")
    log("=" * 70)
    log(f"📊 Resolução final: {RESOLUCAO_FINAL}m")
    log(f"📊 Upscaling total: {fator_total}x (SEN2SR 4x + Swin2SR 2x)")
    log(f"📊 Imagens fusionadas: {len(resultados_temporais)}")
    log(f"📊 Área: {area_km2:.2f} km²")
    log(f"📊 Tempo total: {total_time:.1f}s ({total_time/60:.1f}min)")
    log(f"📊 Modelos: 100% PRÉ-TREINADOS")
    log(f"   - SEN2SR: Treinado para Sentinel-2")
    log(f"   - Swin2SR 2x: Treinado para imagens reais")
    log(f"📂 Resultados em: {OUTPUT_DIR}/")
    log("")
    log("💡 NOTA TÉCNICA:")
    log("   1.25m é o limite realista com modelos prontos.")
    log("   Abaixo disso (0.5m) requereria treinamento específico")
    log("   para Sentinel-2, o que não existe pronto.")
    log("=" * 70)
    
    plt.show()

if __name__ == "__main__":
    main()
