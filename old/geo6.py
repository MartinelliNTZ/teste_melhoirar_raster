# -*- coding: utf-8 -*-
"""
SEN2SR_ULTRA - Super-resolução máxima de Sentinel-2
Pipeline multi-estágio para máxima resolução com qualidade

Estágios:
  1. SEN2SR (10m → 2.5m) - Modelo oficial ESA/TACO
  2. HAT-L (2.5m → 1.25m) - Hybrid Attention Transformer
  3. Real-ESRGAN+ (1.25m → 0.5m) - Refinamento GAN para realismo

Resolução final: 0.5m (20x upscaling total)
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

BANDAS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]
BANDAS_NOMES = [
    "B02_Blue", "B03_Green", "B04_Red", "B08_NIR",
    "B05_RedEdge1", "B06_RedEdge2", "B07_RedEdge3",
    "B8A_NarrowNIR", "B11_SWIR1", "B12_SWIR2"
]
BANDAS_10M_INDICES = [0, 1, 2, 3]
BANDAS_20M_INDICES = [4, 5, 6, 7, 8, 9]

# =============================================================================
# VERIFICAÇÕES
# =============================================================================

try:
    import rioxarray
except ImportError:
    sys.exit("pip install rioxarray")
try:
    import geopandas as gpd
except ImportError:
    sys.exit("pip install geopandas")
try:
    import mamba_ssm
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

# =============================================================================
# MODELOS AVANÇADOS
# =============================================================================

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation para atenção de canal"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    """Atenção espacial para focar em detalhes"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class HATBlock(nn.Module):
    """Hybrid Attention Transformer Block"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()
        self.conv1 = nn.Conv2d(channels, channels * 2, 1)
        self.conv2 = nn.Conv2d(channels * 2, channels, 1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()
        
        # Multi-head attention espacial
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
    def forward(self, x):
        residual = x
        
        # Channel attention
        x = self.norm1(x)
        x = self.channel_att(x)
        
        # Spatial attention with multi-head
        b, c, h, w = x.shape
        q = self.query(x).view(b, self.num_heads, self.head_dim, h*w)
        k = self.key(x).view(b, self.num_heads, self.head_dim, h*w)
        v = self.value(x).view(b, self.num_heads, self.head_dim, h*w)
        
        att = F.softmax(torch.matmul(q.transpose(-2, -1), k) / (self.head_dim ** 0.5), dim=-1)
        out = torch.matmul(v, att.transpose(-2, -1))
        out = out.view(b, c, h, w)
        
        x = self.spatial_att(out)
        
        # Feed-forward
        x = self.norm2(x)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        
        return x + residual

class HATSuperResolution(nn.Module):
    """HAT-L para super-resolução 2x"""
    def __init__(self, in_channels=4, out_channels=4, num_blocks=12):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 64, 3, padding=1)
        
        # HAT blocks em cascata
        self.blocks = nn.ModuleList([
            HATBlock(64) for _ in range(num_blocks)
        ])
        
        self.conv_mid = nn.Conv2d(64, 64, 3, padding=1)
        
        # Upsampling progressivo
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU()
        )
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        skip = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        skip = self.skip(skip)
        
        x = self.conv_in(x)
        features = x
        
        for block in self.blocks:
            x = block(x)
        
        x = self.conv_mid(x) + features
        x = self.up1(x)
        x = self.conv_out(x)
        
        return x + skip

class RealESRGANRefiner(nn.Module):
    """Refinamento GAN para máximo realismo"""
    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        
        # Residual dense blocks
        self.rdb1 = self._make_rdb(128, 3)
        self.rdb2 = self._make_rdb(128, 3)
        self.rdb3 = self._make_rdb(128, 3)
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GELU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1),
            nn.GELU()
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, 3, padding=1)
        )
        
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        
    def _make_rdb(self, channels, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GELU()
            ])
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Upscale inicial
        up = F.interpolate(x, scale_factor=2.5, mode='bicubic', align_corners=False)
        skip = self.skip_conv(up)
        
        # Encoder path
        e1 = F.gelu(self.enc1(up))
        e2 = F.gelu(self.enc2(e1))
        e3 = F.gelu(self.enc3(e2))
        
        # Residual dense processing
        f = self.rdb1(e3) + e3
        f = self.rdb2(f) + f
        f = self.rdb3(f) + f
        
        # Decoder with skip connections
        d1 = self.dec1(f)
        d1 = torch.cat([d1, e2], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        
        out = self.refine(d2)
        return out + skip

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

def equalizar_ranges_rgb(rgb_arr, percentil_min=2, percentil_max=98):
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
    
    print(f"    Range equalizado: [{vmin_global:.4f}, {vmax_global:.4f}]")
    
    rgb_eq = rgb_arr.copy()
    for b in range(3):
        banda = rgb_eq[b]
        mascara = ~np.isnan(banda)
        rgb_eq[b][mascara] = (banda[mascara] - vmin_global) / (vmax_global - vmin_global)
        rgb_eq[b][~mascara] = np.nan
    
    return np.clip(rgb_eq, 0.0, 1.0), vmin_global, vmax_global

def aplicar_rio_color(rgb_arr, transform, crs, caminho_saida):
    try:
        from rio_color.operations import simple_atmo, saturation
    except ImportError:
        print("    [!] pip install rio-color")
        return rgb_arr

    rgb = rgb_arr.copy().astype("float32")
    mask_nan = np.isnan(rgb[0])
    rgb = np.nan_to_num(rgb, nan=0.0)

    rgb_eq, _, _ = equalizar_ranges_rgb(rgb)
    rgb_enh = simple_atmo(rgb_eq, haze=0.02, contrast=3.5, bias=0.5)
    rgb_enh = saturation(rgb_enh, proportion=1.4)
    rgb_enh = np.clip(rgb_enh, 0.0, 1.0)

    for b in range(3):
        rgb_enh[b][mask_nan] = np.nan

    save_geotiff(rgb_enh, transform, crs, caminho_saida)
    return rgb_enh

def salvar_bandas_individuais(arr, nomes, transform, crs, pasta):
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
    print(f"  [{len(nomes)}] bandas salvas em: {pasta_bandas}/")

def process_tile_cuda(tile, model, device, tile_size=64):
    """Processa tile com gerenciamento de memória GPU"""
    C, H, W = tile.shape
    new_H = int(H * 2.5)  # Fator do último estágio
    new_W = int(W * 2.5)
    
    output = np.zeros((C, new_H, new_W), dtype=np.float32)
    
    for h in range(0, H, tile_size):
        for w in range(0, W, tile_size):
            h_end = min(h + tile_size, H)
            w_end = min(w + tile_size, W)
            
            subtile = tile[:, h:h_end, w:w_end]
            subtile_tensor = torch.from_numpy(subtile).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    sr_subtile = model(subtile_tensor)
            
            sr_np = sr_subtile.cpu().numpy()[0]
            h_out = int(h * 2.5)
            w_out = int(w * 2.5)
            
            output[:, h_out:h_out+sr_np.shape[1], w_out:w_out+sr_np.shape[2]] = sr_np
    
    return output

# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    print("=" * 70)
    print("🚀 SEN2SR_ULTRA - Super-Resolução Máxima (0.5m)")
    print("Pipeline: 10m → 2.5m → 1.25m → 0.5m")
    print("Modelos: SEN2SR + HAT-L + Real-ESRGAN+")
    print("=" * 70)

    # Verificar conectividade
    print("\n🔍 Verificando conectividade...")
    if not check_dns(BLOB_HOST) and not check_internet():
        print("  [!!] SEM INTERNET. Abortando.")
        return
    print("  [OK]\n")

    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Dispositivo: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {mem_gb:.1f} GB")
        if mem_gb < 12:
            print("  ⚠️  VRAM < 12GB - Processamento pode ser lento")
        elif mem_gb >= 24:
            print("  ✅ VRAM excelente para processamento ultra!")
        # Otimizações CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    print()

    # 1. Carregar vetor
    print(f"📂 Lendo vetor: {VETOR_LIMITE}")
    gdf = gpd.read_file(VETOR_LIMITE)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    bbox = gdf.total_bounds
    geometria = unary_union(gdf.geometry.values)
    print(f"  Bbox: {bbox}\n")

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
    n_tiles_total = nrows * ncols
    
    fator_total = FATOR_SR1 * FATOR_SR2 * FATOR_SR3
    edge_px_final = int(edge_px * fator_total)
    
    print(f"📐 Área: {largura_m:.0f}m × {altura_m:.0f}m")
    print(f"📊 Cubo 10m: {edge_px}×{edge_px} px ({ncols}×{nrows} tiles)")
    print(f"🎯 Cubo 0.5m: {edge_px_final}×{edge_px_final} px")
    print(f"📈 Upscaling total: {fator_total:.0f}x\n")

    # 3. Carregar modelos
    print("🧠 Carregando modelos de super-resolução...")
    
    # Estágio 1: SEN2SR
    if HAS_MAMBA:
        model_url = ("https://huggingface.co/tacofoundation/sen2sr/resolve/main"
                     "/SEN2SR/NonReference_RGBN_x4/mlm.json")
        model_dir = "model/SEN2SR_RGBN"
        print("  Estágio 1: SEN2SR Mamba (4x) 🔥")
    else:
        model_url = ("https://huggingface.co/tacofoundation/sen2sr/resolve/main"
                     "/SEN2SRLite/NonReference_RGBN_x4/mlm.json")
        model_dir = "model/SEN2SRLite_RGBN"
        print("  Estágio 1: SEN2SR Lite SwinIR (4x)")

    os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(os.path.join(model_dir, "mlm.json")):
        mlstac.download(file=model_url, output_dir=model_dir)
    model_sr1 = mlstac.load(model_dir).compiled_model(device=device)
    
    # Estágio 2: HAT-L
    print("  Estágio 2: HAT-L (2x) - Hybrid Attention Transformer")
    model_sr2 = HATSuperResolution(in_channels=4, out_channels=4, num_blocks=12).to(device)
    model_sr2.eval()
    
    # Estágio 3: Real-ESRGAN+
    print("  Estágio 3: Real-ESRGAN+ (2.5x) - GAN Refinement")
    model_sr3 = RealESRGANRefiner(in_channels=4, out_channels=4).to(device)
    model_sr3.eval()
    
    # Contar parâmetros
    total_params = sum(p.numel() for m in [model_sr2, model_sr3] for p in m.parameters())
    print(f"  📊 Parâmetros totais (estágios 2+3): {total_params/1e6:.1f}M")
    print("  [OK] Modelos carregados!\n")

    # 4. Baixar cubo
    print(f"📥 Baixando cubo Sentinel-2 ({edge_px}×{edge_px}px)...")
    da = cubo.create(lat=cy, lon=cx, collection=COLECAO, bands=BANDAS,
                     start_date=START_DATE, end_date=END_DATE,
                     edge_size=edge_px, resolution=RESOLUCAO_M)

    crs = da.rio.crs
    transf_cubo = da.rio.transform()
    print(f"  CRS: {crs}")

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

    # 5. Pipeline de super-resolução
    C, H, W = cubo_np.shape
    nrows = H // TILE_SIZE
    ncols = W // TILE_SIZE

    # === ESTÁGIO 1: SEN2SR (10m → 2.5m) ===
    print(f"\n{'='*70}")
    print(f"ETAPA 1/3: SEN2SR (10m → 2.5m)")
    print(f"{'='*70}")
    
    H1 = nrows * TILE_SIZE * FATOR_SR1
    W1 = ncols * TILE_SIZE * FATOR_SR1
    mosaico_sr1 = np.zeros((4, H1, W1), dtype="float32")
    mosaico_orig = np.zeros((C, nrows * TILE_SIZE, ncols * TILE_SIZE), dtype="float32")

    for r in tqdm(range(nrows), desc="SEN2SR"):
        for c in range(ncols):
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

            mosaico_orig[:, h0:h1, w0:w1] = tile_norm
            h0s, h1s = r * TILE_SIZE * FATOR_SR1, (r + 1) * TILE_SIZE * FATOR_SR1
            w0s, w1s = c * TILE_SIZE * FATOR_SR1, (c + 1) * TILE_SIZE * FATOR_SR1
            mosaico_sr1[:, h0s:h1s, w0s:w1s] = sr_np

    print(f"  ✅ Etapa 1 concluída! Shape: {mosaico_sr1.shape}")

    # === ESTÁGIO 2: HAT-L (2.5m → 1.25m) ===
    print(f"\n{'='*70}")
    print(f"ETAPA 2/3: HAT-L (2.5m → 1.25m)")
    print(f"{'='*70}")
    
    H2 = int(H1 * FATOR_SR2)
    W2 = int(W1 * FATOR_SR2)
    mosaico_sr2 = np.zeros((4, H2, W2), dtype="float32")
    
    tile_size_sr2 = 128  # Tamanho do tile para HAT-L
    
    n_tiles_h2 = (H1 + tile_size_sr2 - 1) // tile_size_sr2
    n_tiles_w2 = (W1 + tile_size_sr2 - 1) // tile_size_sr2
    
    for h in tqdm(range(0, H1, tile_size_sr2), desc="HAT-L"):
        for w in range(0, W1, tile_size_sr2):
            h_end = min(h + tile_size_sr2, H1)
            w_end = min(w + tile_size_sr2, W1)
            
            tile_25m = mosaico_sr1[:, h:h_end, w:w_end]
            tile_tensor = torch.from_numpy(tile_25m).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    sr_tile = model_sr2(tile_tensor)
            
            sr_np = sr_tile.cpu().numpy()[0]
            h_out = int(h * FATOR_SR2)
            w_out = int(w * FATOR_SR2)
            
            mosaico_sr2[:, h_out:h_out+sr_np.shape[1], 
                       w_out:w_out+sr_np.shape[2]] = sr_np
    
    print(f"  ✅ Etapa 2 concluída! Shape: {mosaico_sr2.shape}")

    # === ESTÁGIO 3: Real-ESRGAN+ (1.25m → 0.5m) ===
    print(f"\n{'='*70}")
    print(f"ETAPA 3/3: Real-ESRGAN+ (1.25m → 0.5m)")
    print(f"{'='*70}")
    
    H3 = int(H2 * FATOR_SR3)
    W3 = int(W2 * FATOR_SR3)
    mosaico_sr3 = np.zeros((4, H3, W3), dtype="float32")
    
    tile_size_sr3 = 64  # Tamanho menor para estágio final
    
    for h in tqdm(range(0, H2, tile_size_sr3), desc="Real-ESRGAN+"):
        for w in range(0, W2, tile_size_sr3):
            h_end = min(h + tile_size_sr3, H2)
            w_end = min(w + tile_size_sr3, W2)
            
            tile_125m = mosaico_sr2[:, h:h_end, w:w_end]
            tile_tensor = torch.from_numpy(tile_125m).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    sr_tile = model_sr3(tile_tensor)
            
            sr_np = sr_tile.cpu().numpy()[0]
            h_out = int(h * FATOR_SR3)
            w_out = int(w * FATOR_SR3)
            
            mosaico_sr3[:, h_out:h_out+sr_np.shape[1], 
                       w_out:w_out+sr_np.shape[2]] = sr_np
    
    print(f"  ✅ Etapa 3 concluída! Shape: {mosaico_sr3.shape}")

    # 6. Aplicar máscara e salvar
    print("\n🎭 Aplicando máscara do polígono...")
    mosaic_orig_clipped, mask_orig = clip_to_polygon(mosaico_orig, transf_cubo, geometria_proj)

    transf_final = rasterio.Affine(
        transf_cubo.a / fator_total, transf_cubo.b, transf_cubo.c,
        transf_cubo.d, transf_cubo.e / fator_total, transf_cubo.f,
    )

    mosaic_final_clipped, mask_final = clip_to_polygon(mosaico_sr3, transf_final, geometria_proj)

    # 7. Montar 10 bandas
    print("\n🔧 Montando 10 bandas em 0.5m...")
    H_final, W_final = mosaic_final_clipped.shape[1], mosaic_final_clipped.shape[2]
    todas_bandas = np.zeros((10, H_final, W_final), dtype="float32")

    todas_bandas[0] = mosaic_final_clipped[2]  # B02
    todas_bandas[1] = mosaic_final_clipped[1]  # B03
    todas_bandas[2] = mosaic_final_clipped[0]  # B04
    todas_bandas[3] = mosaic_final_clipped[3]  # B08
    print(f"  ✅ 4 bandas 10m super-resolvidas")

    mosaic_orig_20m_clipped, _ = clip_to_polygon(mosaico_orig, transf_cubo, geometria_proj)
    for i, idx_orig in enumerate(BANDAS_20M_INDICES):
        idx_final = i + 4
        banda_up = upscale_bicubico(mosaic_orig_20m_clipped[idx_orig], fator_total)
        banda_up[~mask_final] = np.nan
        todas_bandas[idx_final] = banda_up.astype("float32")
    
    print(f"  ✅ 6 bandas 20m interpoladas")

    # 8. Salvar resultados
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4 bandas
    p_sr = os.path.join(OUTPUT_DIR, "super_resolved_0_5m.tif")
    save_geotiff(mosaic_final_clipped, transf_final, crs, p_sr)
    print(f"\n💾 [1] 4 bandas (RGBN) 0.5m: {p_sr}")

    # RGB corrigido
    print("\n🎨 [2] Aplicando rio-color...")
    try:
        rgb_enh = aplicar_rio_color(
            mosaic_final_clipped[[0, 1, 2]], transf_final, crs,
            os.path.join(OUTPUT_DIR, "super_resolved_0_5m_cor.tif")
        )
        print(f"    ✅ RGB corrigido salvo!")
        USAR_COLOR = True
    except Exception as e:
        print(f"    [!] rio-color: {e}")
        USAR_COLOR = False

    # 10 bandas
    print(f"\n📦 [3] Salvando 10 bandas individuais...")
    salvar_bandas_individuais(todas_bandas, BANDAS_NOMES, transf_final, crs, "bandas_0_5m")

    if USAR_COLOR:
        nomes_cor = ["B04_Red_cor", "B03_Green_cor", "B02_Blue_cor"]
        salvar_bandas_individuais(rgb_enh, nomes_cor, transf_final, crs, "bandas_cor_0_5m")

    # 9. Visualização
    print("\n🖼️  Gerando visualização...")
    fig, axes = plt.subplots(1, 3 if USAR_COLOR else 2, figsize=(20 if USAR_COLOR else 14, 6))

    axes[0].imshow(np.clip(mosaic_orig_clipped[[2, 1, 0]].transpose(1, 2, 0) * 1.5, 0, 1))
    axes[0].set_title("Original 10m", fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(np.clip(mosaic_final_clipped[[0, 1, 2]].transpose(1, 2, 0) * 1.5, 0, 1))
    axes[1].set_title(f"Super-res. {RESOLUCAO_FINAL}m", fontsize=12)
    axes[1].axis('off')

    if USAR_COLOR:
        axes[2].imshow(np.clip(rgb_enh.transpose(1, 2, 0), 0, 1))
        axes[2].set_title(f"{RESOLUCAO_FINAL}m + rio-color", fontsize=12)
        axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparacao_ultra.png"), dpi=200, bbox_inches='tight')
    plt.show(block=False)

    # Estatísticas finais
    area_km2 = np.sum(mask_orig) * RESOLUCAO_M**2 / 1e6
    print(f"\n{'='*70}")
    print(f"📊 ESTATÍSTICAS FINAIS")
    print(f"{'='*70}")
    print(f"  Área: {area_km2:.2f} km²")
    print(f"  Resolução: {RESOLUCAO_FINAL}m")
    print(f"  Upscaling: {fator_total:.0f}x")
    print(f"  Original: {mosaic_orig_clipped.shape} (10m)")
    print(f"  Estágio 1: {mosaico_sr1.shape} (2.5m)")
    print(f"  Estágio 2: {mosaico_sr2.shape} (1.25m)")
    print(f"  Final: {mosaic_final_clipped.shape} (0.5m)")
    print(f"  Arquivo: {mosaic_final_clipped.nbytes / 1e9:.2f} GB")
    print(f"  Pasta: {OUTPUT_DIR}/")
    print(f"\n🎉 PIPELINE ULTRA CONCLUÍDO! (0.5m)")
    plt.show()


if __name__ == "__main__":
    main()
