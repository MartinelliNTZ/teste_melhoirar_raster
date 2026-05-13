# -*- coding: utf-8 -*-
"""
SEN2SR_ULTRA_V2 - Super-resolução máxima com modelos PRÉ-TREINADOS
Pipeline: 10m → 2.5m (SEN2SR) → 0.5m (Swin2SR x4)
+ Fusão multi-temporal para melhoria de qualidade
CORRIGIDO: busca por data com janela
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
from rasterio.warp import reproject, Resampling
from shapely.ops import unary_union
from scipy.ndimage import zoom as scipy_zoom
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
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
START_DATE   = "2026-04-08"  # Data original que funcionava
END_DATE     = "2026-05-07"
MAX_IMAGES   = 3             # Número de imagens para fusão temporal

COLECAO      = "sentinel-2-l2a"
RESOLUCAO_M  = 10
TILE_SIZE    = 128
OUTPUT_DIR   = "resultados_ultra_v2"
BLOB_HOST    = "sentinel2l2a01.blob.core.windows.net"

# Pipeline com modelos PRÉ-TREINADOS
FATOR_SR1    = 4        # 10m → 2.5m (SEN2SR - treinado)
FATOR_SR2    = 5        # 2.5m → 0.5m (Swin2SR x4 real + refinamento)
RESOLUCAO_FINAL = 0.5   # Resolução alvo em metros

# Otimizações
BATCH_SIZE = 4
TILE_OVERLAP = 16
USE_AMP = False          # Desabilitado para evitar artefatos

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
# MODELOS PRÉ-TREINADOS (Swin2SR via HuggingFace)
# =============================================================================

class Swin2SRSuperResolution:
    """Wrapper para Swin2SR pré-treinado (modelo real-world SR)"""
    def __init__(self, device, scale=4):
        self.device = device
        self.scale = scale
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Carrega Swin2SR do HuggingFace"""
        try:
            from transformers import Swin2SRForImageSuperResolution
            from transformers import Swin2SRImageProcessor
            
            log("Carregando Swin2SR pré-treinado (Real-World SR)...")
            
            model_id = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
            
            self.processor = Swin2SRImageProcessor.from_pretrained(model_id)
            self.model = Swin2SRForImageSuperResolution.from_pretrained(
                model_id,
                torch_dtype=torch.float32
            ).to(self.device)
            self.model.eval()
            
            log(f"✅ Swin2SR carregado! Parâmetros: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M")
            
        except ImportError:
            log("Transformers não instalado. Tentando alternativa...", "WARN")
            self._load_alternative()
    
    def _load_alternative(self):
        """Alternativa: ESRGAN pré-treinado"""
        try:
            import esrgan_pytorch
            
            log("Carregando ESRGAN pré-treinado...")
            self.model = esrgan_pytorch.ESRGAN(scale=4, device=self.device)
            self.model.load_pretrained()
            self.model.eval()
            log("✅ ESRGAN carregado!")
            
        except ImportError:
            log("Nenhum modelo pré-treinado disponível!", "ERROR")
            log("Instale: pip install transformers timm", "ERROR")
            log("Ou: pip install esrgan-pytorch", "ERROR")
            raise
    
    def __call__(self, x):
        """Processa tensor ou numpy array"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            if hasattr(self, 'processor'):
                # Swin2SR pipeline supports RGB only
                if x.max() > 1.0:
                    x = x / 255.0
                
                use_nir = x.shape[1] == 4
                if use_nir:
                    rgb = x[:, :3, :, :]
                    nir = x[:, 3:4, :, :]
                else:
                    rgb = x
                    nir = None
                
                outputs = []
                for i in range(rgb.shape[0]):
                    img = rgb[i].cpu().numpy().transpose(1, 2, 0)
                    img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
                    
                    inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                    output = self.model(**inputs)
                    output = output.reconstruction.squeeze(0).cpu().numpy()
                    output = output.transpose(2, 0, 1) / 255.0
                    outputs.append(torch.from_numpy(output))
                
                sr_rgb = torch.stack(outputs).to(self.device)
                if nir is not None:
                    # Upscale NIR using bicubic to match Swin2SR output size
                    sr_nir = F.interpolate(
                        nir,
                        size=(sr_rgb.shape[2], sr_rgb.shape[3]),
                        mode='bicubic',
                        align_corners=False
                    )
                    return torch.cat([sr_rgb, sr_nir], dim=1)
                return sr_rgb
            else:
                return self.model(x)
    
    def to(self, device):
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self

# =============================================================================
# REFINAMENTO LEVE
# =============================================================================

class LightRefiner(nn.Module):
    """Refinamento leve sem treinamento - apenas melhoria de bordas"""
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        
        with torch.no_grad():
            kernel = torch.tensor([[[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]]], 
                                dtype=torch.float32)
            for i in range(in_channels):
                self.conv1.weight.data[i:i+1] = kernel
                self.conv2.weight.data[i:i+1] = kernel * 0.5
            
            self.conv1.bias.data.zero_()
            self.conv2.bias.data.zero_()
    
    def forward(self, x):
        edge = self.conv1(x)
        smooth = self.conv2(x)
        return x + 0.1 * (edge - smooth)

# =============================================================================
# PROCESSAMENTO MULTI-TEMPORAL
# =============================================================================

class MultiTemporalFusion:
    """Fusão de múltiplas imagens temporais para melhor qualidade"""
    
    @staticmethod
    def compute_quality_score(image):
        """Calcula score de qualidade para cada pixel/imagem"""
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
        
        if method == 'mean':
            return np.nanmean(images_stack, axis=0)
        
        elif method == 'median':
            return np.nanmedian(images_stack, axis=0)
        
        elif method == 'weighted_median':
            weights = []
            for img in images:
                score = MultiTemporalFusion.compute_quality_score(img)
                weights.append(score)
            
            weights = np.stack(weights, axis=0)
            weights = weights / (weights.sum(axis=0, keepdims=True) + 1e-8)
            
            fused = np.sum(images_stack * weights[:, np.newaxis, :, :], axis=0)
            return fused
        
        elif method == 'max_sharpness':
            sharpness = np.zeros(len(images))
            for i, img in enumerate(images):
                sharpness[i] = MultiTemporalFusion.compute_quality_score(img).mean()
            
            best_idx = np.argmax(sharpness)
            log(f"   Melhor imagem: índice {best_idx} (score: {sharpness[best_idx]:.4f})")
            return images[best_idx]
        
        else:
            return np.nanmean(images_stack, axis=0)

# =============================================================================
# FUNÇÕES DE PROCESSAMENTO
# =============================================================================

def apply_blend_window(tile_size, overlap):
    """Janela de blending para evitar artefatos"""
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

# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def baixar_cubo_temporal(cy, cx, edge_px, start_date, end_date, max_images=3):
    """
    Baixa múltiplas imagens em diferentes períodos dentro da janela
    CORRIGIDO: usa janela de 5 dias ao redor de cada data alvo
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days
    
    if total_days <= 0:
        # Se for data única, usa a data original que funcionava
        log("Janela muito curta, usando data original...", "WARN")
        return baixar_cubo_unico(cy, cx, edge_px, start_date, end_date)
    
    # Dividir o período em intervalos
    interval = max(total_days // max_images, 10)  # Mínimo 10 dias entre imagens
    
    cubos_temporais = []
    crs = None
    transf_cubo = None
    
    for i in range(max_images):
        # Data central do intervalo
        center_date = start_dt + timedelta(days=i * interval + interval // 2)
        if center_date > end_dt:
            center_date = end_dt
        
        # Janela de 5 dias ao redor
        sub_start = (center_date - timedelta(days=2)).strftime("%Y-%m-%d")
        sub_end = (center_date + timedelta(days=2)).strftime("%Y-%m-%d")
        
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
            
            log(f"   ✅ Imagem {i+1} baixada! Shape: {cubo_np.shape}")
            
        except Exception as e:
            log(f"   ⚠️ Falha ao baixar imagem {i+1}: {e}", "WARN")
            continue
    
    return cubos_temporais, crs, transf_cubo

def baixar_cubo_unico(cy, cx, edge_px, start_date, end_date):
    """Fallback: baixa uma única imagem"""
    log(f"Baixando imagem única: {start_date} a {end_date}")
    
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
    
    log(f"✅ Imagem baixada! Shape: {cubo_np.shape}")
    
    return [cubo_norm], crs, transf_cubo

def main():
    start_time = time.time()
    log("=" * 70)
    log("SEN2SR_ULTRA_V2 - Super-Resolução com Modelos PRÉ-TREINADOS")
    log("Pipeline: 10m → 2.5m (SEN2SR) → 0.5m (Swin2SR/ESRGAN)")
    log(f"Multi-temporal: até {MAX_IMAGES} imagens")
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
    
    fator_total = FATOR_SR1 * FATOR_SR2
    edge_px_final = int(edge_px * fator_total)
    
    log(f"Área: {largura_m:.0f}m × {altura_m:.0f}m")
    log(f"Cubo 10m: {edge_px}×{edge_px} px")
    log(f"Cubo 0.5m: {edge_px_final}×{edge_px_final} px")

    # 3. Carregar modelos PRÉ-TREINADOS
    log("=" * 70)
    log("CARREGANDO MODELOS PRÉ-TREINADOS")
    log("=" * 70)
    
    # Estágio 1: SEN2SR (treinado para Sentinel-2)
    try:
        import mamba_ssm
        model_url = ("https://huggingface.co/tacofoundation/sen2sr/resolve/main"
                     "/SEN2SR/NonReference_RGBN_x4/mlm.json")
        model_dir = "model/SEN2SR_RGBN"
        log("✅ Estágio 1: SEN2SR Mamba (4x) - TREINADO para Sentinel-2")
    except ImportError:
        model_url = ("https://huggingface.co/tacofoundation/sen2sr/resolve/main"
                     "/SEN2SRLite/NonReference_RGBN_x4/mlm.json")
        model_dir = "model/SEN2SRLite_RGBN"
        log("✅ Estágio 1: SEN2SR Lite SwinIR (4x) - TREINADO para Sentinel-2")

    os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(os.path.join(model_dir, "mlm.json")):
        mlstac.download(file=model_url, output_dir=model_dir)
    model_sr1 = mlstac.load(model_dir).compiled_model(device=device)
    
    # Estágio 2: Swin2SR ou ESRGAN (pré-treinado para imagens reais)
    log("Carregando Estágio 2: Modelo SR pré-treinado...")
    
    try:
        model_sr2 = Swin2SRSuperResolution(device, scale=4)
        log("✅ Estágio 2: Swin2SR (4x) - PRÉ-TREINADO em imagens reais")
        USE_SWIN = True
    except Exception as e:
        log(f"Swin2SR não disponível: {e}", "WARN")
        log("Usando bicubic + refinamento leve...")
        USE_SWIN = False
        model_sr2 = LightRefiner(in_channels=4).to(device)
        model_sr2.eval()
        log("⚠️ Estágio 2: Bicubic + LightRefiner (qualidade limitada)")
    
    # Refinamento final leve
    refiner = LightRefiner(in_channels=4).to(device)
    refiner.eval()
    
    log("=" * 70)

    # 4. Baixar imagens (CORRIGIDO)
    log("Baixando imagens...")
    
    cubos_temporais, crs, transf_cubo = baixar_cubo_temporal(
        cy, cx, edge_px, START_DATE, END_DATE, MAX_IMAGES
    )
    
    if len(cubos_temporais) == 0:
        log("Nenhuma imagem disponível! Abortando.", "ERROR")
        return
    
    log(f"✅ {len(cubos_temporais)} imagens baixadas com sucesso")
    
    # Reprojetar polígono
    from shapely.ops import transform as shapely_transform
    project = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    geometria_proj = shapely_transform(project, geometria)
    
    C, H, W = cubos_temporais[0].shape

    # 5. Processar cada imagem temporal
    log("=" * 70)
    log("PROCESSANDO IMAGENS")
    log("=" * 70)
    
    resultados_temporais = []
    
    for img_idx, cubo_norm in enumerate(cubos_temporais):
        log(f"\n{'='*50}")
        log(f"Processando imagem {img_idx+1}/{len(cubos_temporais)}")
        log(f"{'='*50}")
        
        # === ESTÁGIO 1: SEN2SR (10m → 2.5m) ===
        log("Etapa 1: SEN2SR (10m → 2.5m)")
        
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
        diagnosticar_tensor(mosaico_sr1, f"Imagem {img_idx+1} - SEN2SR")
        
        # === ESTÁGIO 2: Swin2SR/ESRGAN ===
        log("Etapa 2: Modelo SR pré-treinado")
        
        if USE_SWIN:
            # Swin2SR processa em tiles
            H2 = int(H1 * 4)
            W2 = int(W1 * 4)
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
            
            for i in tqdm(range(0, len(tiles_list), BATCH_SIZE), desc="Swin2SR"):
                batch_positions = tiles_list[i:i+BATCH_SIZE]
                
                batch_tiles = []
                for h, w, h_end, w_end in batch_positions:
                    tile = mosaico_sr1[:, h:h_end, w:w_end]
                    # Padroniza para (4, 64, 64) usando padding/reflexão
                    pad_h = tile_size_sr2 - tile.shape[1]
                    pad_w = tile_size_sr2 - tile.shape[2]
                    if pad_h > 0 or pad_w > 0:
                        padded = np.pad(
                            tile,
                            ((0, 0), (0, pad_h), (0, pad_w)),
                            mode='reflect'
                        )
                        batch_tiles.append(padded)
                    else:
                        batch_tiles.append(tile)
                
                # Envia apenas os 3 primeiros canais (RGB) para o Swin2SR
                batch_tensor = torch.stack([torch.from_numpy(t[:3, :, :]).float() for t in batch_tiles]).to(device)
                with torch.no_grad():
                    sr_tiles = model_sr2(batch_tensor).cpu().numpy()
                
                for idx, (h, w, h_end, w_end) in enumerate(batch_positions):


                        sr_tile = sr_tiles[idx]
                        log(f"Shape original sr_tile: {sr_tile.shape}", "WARN")
                        tile_h = int((h_end - h) * 4)
                        tile_w = int((w_end - w) * 4)

                        # Estratégias para garantir shape correto
                        success = False
                        formas = []
                        try:
                            # 1. Caso comum
                            if sr_tile.shape in [(3, tile_h, tile_w), (4, tile_h, tile_w)]:
                                formas.append('direto')
                                success = True
                            # 2. (tile_h, tile_w, 3) ou (tile_h, tile_w, 4)
                            elif sr_tile.shape in [(tile_h, tile_w, 3), (tile_h, tile_w, 4)]:
                                sr_tile = sr_tile.transpose(2, 0, 1)
                                formas.append('transpose(2,0,1)')
                                success = True
                            # 3. (N, 3, tile_h, tile_w) ou (N, 4, tile_h, tile_w)
                            elif len(sr_tile.shape) == 4 and sr_tile.shape[1] in [3, 4]:
                                sr_tile = sr_tile[0]
                                formas.append('batch[0]')
                                success = True
                            # 4. (C*tile_h, tile_w)
                            elif sr_tile.shape[1] == tile_w and sr_tile.shape[0] % tile_h == 0:
                                c = sr_tile.shape[0] // tile_h
                                sr_tile = sr_tile.reshape((c, tile_h, tile_w))
                                formas.append('reshape')
                                success = True
                            # 5. (tile_h*tile_w*3,)
                            elif sr_tile.size == 3*tile_h*tile_w:
                                sr_tile = sr_tile.reshape((3, tile_h, tile_w))
                                formas.append('reshape_flat3')
                                success = True
                            # 6. (tile_h*tile_w*4,)
                            elif sr_tile.size == 4*tile_h*tile_w:
                                sr_tile = sr_tile.reshape((4, tile_h, tile_w))
                                formas.append('reshape_flat4')
                                success = True
                            # 7. Crop automático se tile maior
                            elif sr_tile.shape[0] in [3, 4] and (sr_tile.shape[1] >= tile_h and sr_tile.shape[2] >= tile_w):
                                sr_tile = sr_tile[:, :tile_h, :tile_w]
                                formas.append('crop_auto')
                                success = True
                            elif sr_tile.shape[1] in [3, 4] and (sr_tile.shape[0] >= tile_h and sr_tile.shape[2] >= tile_w):
                                sr_tile = sr_tile.transpose(1, 0, 2)[:, :tile_h, :tile_w]
                                formas.append('crop_auto_transpose')
                                success = True
                        except Exception as e:
                            log(f"Falha ao tentar corrigir shape: {e}", "ERROR")
                            success = False

                        if not success:
                            log(f"Shape sr_tile impossível de corrigir automaticamente: {sr_tile.shape}", "ERROR")
                            raise ValueError(f"Shape sr_tile impossível de corrigir automaticamente: {sr_tile.shape}, tile_h={tile_h}, tile_w={tile_w}")
                        else:
                            log(f"Shape sr_tile corrigido com: {formas}", "INFO")

                        # If only 3 channels, upsample NIR from input tile
                        if sr_tile.shape[0] == 3:
                            nir_in = batch_tiles[idx][3:4, :, :]
                            nir_up = scipy_zoom(nir_in, (1, tile_h / nir_in.shape[1], tile_w / nir_in.shape[2]), order=3)
                            nir_up = np.squeeze(nir_up, axis=0)
                            sr_tile = np.concatenate([sr_tile, nir_up[None, ...]], axis=0)

                        # --- DEBUG: log shapes ---
                        if sr_tile.shape != (4, tile_h, tile_w):
                            log(f"Corrigindo shape sr_tile: {sr_tile.shape} para (4, {tile_h}, {tile_w})", "WARN")
                            try:
                                sr_tile = sr_tile.reshape((4, tile_h, tile_w))
                            except Exception as e:
                                log(f"Falha ao reshape: {e}", "ERROR")
                                raise

                        h_out = int(h * 4)
                        w_out = int(w * 4)

                        window_resized = scipy_zoom(blend_window, (tile_h/tile_size_sr2, tile_w/tile_size_sr2), order=1)

                        for ch in range(4):
                            canal = sr_tile[ch]
                            if canal.shape != (tile_h, tile_w):
                                canal = canal.reshape((tile_h, tile_w))
                            mosaico_sr2[ch, h_out:h_out+tile_h, w_out:w_out+tile_w] += canal * window_resized
                        weight[h_out:h_out+tile_h, w_out:w_out+tile_w] += window_resized
            
            weight[weight == 0] = 1
            for ch in range(4):
                mosaico_sr2[ch] /= weight
            
            # Upscale de 0.625m para 0.5m
            scale_625_to_500 = 0.625 / 0.5
            H_final = int(H2 * scale_625_to_500)
            W_final = int(W2 * scale_625_to_500)
            
            mosaico_final = np.zeros((4, H_final, W_final), dtype=np.float32)
            for ch in range(4):
                mosaico_final[ch] = scipy_zoom(mosaico_sr2[ch], scale_625_to_500, order=3)
            
        else:
            # Fallback: bicubic 5x + refinamento
            H_final = int(H1 * 5)
            W_final = int(W1 * 5)
            
            mosaico_final = np.zeros((4, H_final, W_final), dtype=np.float32)
            for ch in range(4):
                mosaico_final[ch] = scipy_zoom(mosaico_sr1[ch], 5.0, order=3)
            
            log("Aplicando refinamento leve...")
            mosaico_final = np.clip(mosaico_final, 0, 1)
            
            tile_size_ref = 128
            for h in range(0, H_final, tile_size_ref):
                for w in range(0, W_final, tile_size_ref):
                    h_end = min(h + tile_size_ref, H_final)
                    w_end = min(w + tile_size_ref, W_final)
                    
                    tile = mosaico_final[:, h:h_end, w:w_end]
                    tile_tensor = torch.from_numpy(tile).float().unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        refined = refiner(tile_tensor).squeeze(0).cpu().numpy()
                    
                    mosaico_final[:, h:h_end, w:w_end] = np.clip(refined, 0, 1)
        
        diagnosticar_tensor(mosaico_final, f"Imagem {img_idx+1} - Final")
        resultados_temporais.append(mosaico_final)
    
    # 6. Fusão temporal
    log("=" * 70)
    log("FUSÃO TEMPORAL")
    log("=" * 70)
    
    if len(resultados_temporais) > 1:
        log(f"Fusionando {len(resultados_temporais)} imagens...")
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
    
    mosaic_orig = cubos_temporais[0][[2, 1, 0, 3], :, :]
    mosaic_orig_clipped, mask_orig = clip_to_polygon(
        mosaic_orig, transf_cubo, geometria_proj
    )
    
    # 8. Salvar resultados
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    p_sr = os.path.join(OUTPUT_DIR, f"super_resolved_{RESOLUCAO_FINAL}m.tif")
    save_geotiff(mosaic_final_clipped, transf_final, crs, p_sr)
    log(f"✅ Resultado salvo: {p_sr}")
    
    # 9. Visualização
    log("Gerando visualização...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(np.clip(mosaic_orig_clipped[[0, 1, 2]].transpose(1, 2, 0) * 2.5, 0, 1))
    axes[0].set_title(f"Original 10m", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    rgb_final = mosaic_final_clipped[[0, 1, 2]]
    axes[1].imshow(np.clip(rgb_final.transpose(1, 2, 0) * 2.5, 0, 1))
    axes[1].set_title(f"Super-Resolvido {RESOLUCAO_FINAL}m\nSwin2SR + Fusão Temporal", 
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparacao_final.png"), dpi=200, bbox_inches='tight')
    log("✅ Visualização salva")
    
    # 10. Estatísticas
    total_time = time.time() - start_time
    area_km2 = np.sum(mask_orig) * RESOLUCAO_M**2 / 1e6
    
    log("=" * 70)
    log("✅ PIPELINE CONCLUÍDO!")
    log("=" * 70)
    log(f"📊 Resolução final: {RESOLUCAO_FINAL}m")
    log(f"📊 Upscaling total: {fator_total:.0f}x")
    log(f"📊 Imagens fusionadas: {len(resultados_temporais)}")
    log(f"📊 Área: {area_km2:.2f} km²")
    log(f"📊 Tempo total: {total_time:.1f}s ({total_time/60:.1f}min)")
    log(f"📊 Modelos: SEN2SR (treinado) + Swin2SR (pré-treinado)")
    log(f"📂 Resultados em: {OUTPUT_DIR}/")
    log("=" * 70)
    
    plt.show()

if __name__ == "__main__":
    main()
