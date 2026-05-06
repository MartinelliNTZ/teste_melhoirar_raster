# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import mlstac
import cubo
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

# Verificação essencial: rioxarray precisa estar instalado
try:
    import rioxarray  # noqa: F401 – necessário para o accessor .rio
except ImportError:
    sys.exit(
        "Erro: o pacote 'rioxarray' não está instalado.\n"
        "Instale-o com: pip install rioxarray"
    )

"""
SEN2SR - Super-resolução de imagens Sentinel-2 para 2.5m
Salva a imagem original e a super-resolvida como GeoTIFFs georreferenciados.
"""

# Tenta verificar se o mamba_ssm está disponível
try:
    import mamba_ssm
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

# Parâmetros da Área de Interesse (AOI)
LATITUDE = -10.18440098
LONGITUDE = -48.33361440
START_DATE = "2024-09-08"
END_DATE = "2025-09-08"
IMAGE_INDEX = 0
OUTPUT_DIR = "resultados"
SCALE_FACTOR = 10

def save_geotiff(array, transform, crs, filepath, dtype=rasterio.float32):
    """Salva array (C, H, W) como GeoTIFF."""
    count, height, width = array.shape
    with rasterio.open(
        filepath,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(array.astype(dtype))

def main():
    # 1. Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 2. Seleção do modelo
    if HAS_MAMBA:
        model_url = "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SR/NonReference_RGBN_x4/mlm.json"
        model_dir = "model/SEN2SR_RGBN"
        print("Usando arquitetura: Mamba (Full)")
    else:
        model_url = "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json"
        model_dir = "model/SEN2SRLite_RGBN"
        print("Usando arquitetura: SwinIR (Lite)")

    os.makedirs(model_dir, exist_ok=True)
    print(f"Baixando pesos do modelo em {model_dir}...")
    mlstac.download(file=model_url, output_dir=model_dir)

    # 3. Cubo Sentinel-2
    print("Criando cubo de dados Sentinel-2 (B04, B03, B02, B08)...")
    da = cubo.create(
        lat=LATITUDE,
        lon=LONGITUDE,
        collection="sentinel-2-l2a",
        bands=["B04", "B03", "B02", "B08"],
        start_date=START_DATE,
        end_date=END_DATE,
        edge_size=128,
        resolution=10
    )

    # Agora .rio está disponível graças ao rioxarray
    original_crs = da.rio.crs
    original_transform = da.rio.transform()
    print(f"CRS: {original_crs}")

    # 4. Pré-processamento
    print("Normalizando imagem...")
    original_s2_numpy = (da[IMAGE_INDEX].compute().to_numpy() / 10000).astype("float32")
    X = torch.from_numpy(original_s2_numpy).float().to(device)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 5. Inferência
    print("Carregando e compilando o modelo...")
    model = mlstac.load(model_dir).compiled_model(device=device)

    print("Executando super-resolução...")
    with torch.no_grad():
        superX = model(X[None]).squeeze(0)
    super_resolved_numpy = superX.cpu().numpy().astype("float32")

    # 6. Salvamento
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Original 10m
    original_path = os.path.join(OUTPUT_DIR, "original_10m.tif")
    save_geotiff(original_s2_numpy, original_transform, original_crs, original_path)
    print(f"Original salvo em: {original_path}")

    # Super-resolvido 2.5m (transformação ajustada)
    new_transform = rasterio.Affine(
        original_transform.a / SCALE_FACTOR,
        original_transform.b,
        original_transform.c,
        original_transform.d,
        original_transform.e / SCALE_FACTOR,
        original_transform.f
    )
    super_path = os.path.join(OUTPUT_DIR, "super_resolved_2_5m.tif")
    save_geotiff(super_resolved_numpy, new_transform, original_crs, super_path)
    print(f"Super-resolvido salvo em: {super_path}")

    # 7. Visualização
    print("Gerando comparação...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(X[[0,1,2]].permute(1,2,0).cpu().numpy() * 1.5)
    axes[0].set_title("Sentinel-2 Original (10m)")
    axes[0].axis('off')
    axes[1].imshow(superX[[0,1,2]].permute(1,2,0).cpu().numpy() * 1.5)
    axes[1].set_title("SEN2SR Super-resolved (2.5m)")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()