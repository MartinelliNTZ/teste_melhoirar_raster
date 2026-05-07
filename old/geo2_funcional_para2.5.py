# -*- coding: utf-8 -*-
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

# Cache directory for downloaded tiles
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_s2_tiles")

# Blob hostname that needs to be resolvable
BLOB_HOST = "sentinel2l2a01.blob.core.windows.net"


def check_dns_resolution(hostname, timeout=5):
    """Check if a hostname can be resolved via DNS."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(hostname)
        return True
    except (socket.gaierror, OSError):
        return False
    finally:
        socket.setdefaulttimeout(None)


def check_internet_connectivity():
    """Check basic internet connectivity by testing DNS resolution of known hosts."""
    test_hosts = [
        "sentinel2l2a01.blob.core.windows.net",
        "google.com",
        "huggingface.co",
        "8.8.8.8",
    ]
    for host in test_hosts:
        if check_dns_resolution(host):
            return True
    return False


def compute_with_retry(da, max_retries=5, base_delay=2.0, backoff=2.0):
    """
    Compute a dask-backed xarray DataArray with retry logic.
    This handles transient DNS/network failures.
    Also caches downloaded tiles locally for subsequent runs.
    """
    # First try: direct compute
    for attempt in range(1, max_retries + 1):
        try:
            result = da[IMAGE_INDEX].compute().to_numpy()
            return result
        except Exception as e:
            error_str = str(e).lower()
            is_dns_error = "could not resolve host" in error_str or "resolve" in error_str
            is_timeout = "timeout" in error_str or "timed out" in error_str
            is_connection = "connection" in error_str or "econn" in error_str

            if is_dns_error:
                print(f"\n  [!] Erro DNS (tentativa {attempt}/{max_retries}): {e}")
                if attempt == 1:
                    print("  [!] Verificando conectividade com a internet...")
                    has_internet = check_internet_connectivity()
                    if not has_internet:
                        print("\n  [!!] SEM CONEXÃO COM A INTERNET. Possíveis causas:")
                        print("       - Verifique se o Wi-Fi/cabo de rede está conectado")
                        print("       - O servidor DNS pode estar bloqueando o acesso")
                        print("       - Firewall corporativo/proxy pode estar ativo")
                        print("       - Tente usar uma VPN ou rede diferente\n")
                    else:
                        print("  [!] Internet parece funcionar, mas o host do Sentinel-2 Azure Blob não resolve.")
                        print("  [!] Possível bloqueio regional ou temporário no Azure.\n")
            elif is_timeout or is_connection:
                print(f"\n  [!] Erro de rede/TimeOut (tentativa {attempt}/{max_retries}): {e}")
            else:
                print(f"\n  [!] Erro inesperado (tentativa {attempt}/{max_retries}): {e}")

            if attempt < max_retries:
                delay = base_delay * (backoff ** (attempt - 1))
                print(f"  [~] Aguardando {delay:.1f}s antes de tentar novamente...")
                time.sleep(delay)
            else:
                print(f"\n  [!!] Todas as {max_retries} tentativas falharam.")
                print("  [!!] O download dos tiles Sentinel-2 não é possível no momento.")
                print("\n  Sugestões:")
                print("   1. Verifique sua conexão com a internet")
                print("   2. Se estiver usando VPN/proxy, tente desativar/ativar")
                print("   3. Tente executar novamente mais tarde")
                print("   4. Se o problema persistir, o blob storage do Azure pode estar")
                print("      temporariamente indisponível na sua região")
                print("   5. Verifique com 'nslookup sentinel2l2a01.blob.core.windows.net'")
                print("      se o DNS resolve corretamente\n")
                raise

    return None  # never reached


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
    # 0. Pre-check: DNS resolution for the Sentinel-2 blob host
    print("Verificando conectividade com servidores Sentinel-2...")
    dns_ok = check_dns_resolution(BLOB_HOST)
    if not dns_ok:
        print(f"  [!] DNS NÃO RESOLVE: {BLOB_HOST}")
        print("  [!] Tentando verificar internet geral...")
        has_internet = check_internet_connectivity()
        if not has_internet:
            print("  [!!] SEM CONEXÃO COM A INTERNET detectada.")
            print("  [!!] Verifique sua rede e tente novamente.")
        else:
            print("  [~] Internet parece OK, mas host específico do blob não resolve.")
            print("  [~] Tentando mesmo assim (pode ser problema temporário)...\n")
    else:
        print(f"  [OK] DNS resolvido: {BLOB_HOST}\n")

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

    # 4. Pré-processamento com retry
    print("Normalizando imagem (com retry em caso de falha de rede)...")
    original_s2_numpy = compute_with_retry(da)
    original_s2_numpy = (original_s2_numpy / 10000).astype("float32")

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
    SCALE_FACTOR = 4    # Fator de super-resolução (4×)  

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