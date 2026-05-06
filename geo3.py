# -*- coding: utf-8 -*-
"""
SEN2SR - Super-resolução de imagens Sentinel-2 para 2.5m
Usa um arquivo vetorial (GPKG) como limite da Área de Interesse (AOI).
Salva a imagem original e a super-resolvida como GeoTIFFs georreferenciados.
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

# =============================================================================
# CONFIGURAÇÕES (centralizadas no início)
# =============================================================================

# Caminho do arquivo vetorial com o limite da AOI (polígono ou multipolígono)
VETOR_LIMITE = "vetores/limite.gpkg"

# Dados temporais
START_DATE = "2024-09-08"
END_DATE   = "2025-09-08"
IMAGE_INDEX = 0  # índice da imagem dentro do cubo temporal

# Configuração do cubo Sentinel-2
COLECAO = "sentinel-2-l2a"
BANDAS  = ["B04", "B03", "B02", "B08"]  # RGB + NIR
RESOLUCAO_M = 10   # resolução original Sentinel-2 L2A (metros)
EDGE_SIZE   = 128  # tamanho do tile (pixels) – usado se bbox não for suportado

# Super-resolução
FATOR_SUPER_RESOLUCAO = 4   # 10m -> 2.5m

# Saída
OUTPUT_DIR = "resultados"

# Host do blob Azure para pré-verificação de DNS
BLOB_HOST = "sentinel2l2a01.blob.core.windows.net"

# =============================================================================
# VERIFICAÇÕES DE DEPENDÊNCIAS
# =============================================================================

# rioxarray (necessário para .rio accessor)
try:
    import rioxarray  # noqa: F401
except ImportError:
    sys.exit("Erro: o pacote 'rioxarray' não está instalado.\n"
             "Instale-o com: pip install rioxarray")

# geopandas (para ler o GPKG vetorial)
try:
    import geopandas as gpd
except ImportError:
    sys.exit("Erro: o pacote 'geopandas' não está instalado.\n"
             "Instale-o com: pip install geopandas")

# mamba_ssm (opcional, para arquitetura Mamba Full)
try:
    import mamba_ssm  # noqa: F401
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def check_dns_resolution(hostname, timeout=5):
    """Verifica se um hostname pode ser resolvido via DNS."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(hostname)
        return True
    except (socket.gaierror, OSError):
        return False
    finally:
        socket.setdefaulttimeout(None)


def check_internet_connectivity():
    """Verifica conectividade básica com a internet."""
    test_hosts = [
        BLOB_HOST,
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
    Faz o compute de um DataArray com retry para falhas de rede/DNS.
    """
    for attempt in range(1, max_retries + 1):
        try:
            result = da[IMAGE_INDEX].compute().to_numpy()
            return result
        except Exception as e:
            error_str = str(e).lower()
            is_dns = "could not resolve host" in error_str or "resolve" in error_str
            is_timeout = "timeout" in error_str or "timed out" in error_str
            is_conn = "connection" in error_str or "econn" in error_str

            if is_dns:
                print(f"\n  [!] Erro DNS (tentativa {attempt}/{max_retries}): {e}")
                if attempt == 1:
                    print("  [!] Verificando conectividade com a internet...")
                    if not check_internet_connectivity():
                        print("\n  [!!] SEM CONEXÃO COM A INTERNET.")
                        print("       Verifique Wi-Fi/cabo, firewall ou proxy.\n")
                    else:
                        print("  [!] Internet funciona, mas host do Azure Blob não resolve.")
                        print("  [!) Possível bloqueio regional ou temporário.\n")
            elif is_timeout or is_conn:
                print(f"\n  [!] Erro de rede/Timeout (tentativa {attempt}/{max_retries}): {e}")
            else:
                print(f"\n  [!] Erro inesperado (tentativa {attempt}/{max_retries}): {e}")

            if attempt < max_retries:
                delay = base_delay * (backoff ** (attempt - 1))
                print(f"  [~] Aguardando {delay:.1f}s antes de tentar novamente...")
                time.sleep(delay)
            else:
                print(f"\n  [!!] Todas as {max_retries} tentativas falharam.")
                print("  [!!] Não foi possível baixar os tiles Sentinel-2.")
                print("\n  Sugestões:")
                print("   1. Verifique sua conexão com a internet")
                print("   2. Desative/ative VPN ou proxy")
                print("   3. Tente novamente mais tarde")
                print(f"   4. Teste com: nslookup {BLOB_HOST}\n")
                raise

    return None


def save_geotiff(array, transform, crs, filepath, dtype=rasterio.float32):
    """Salva um array (C, H, W) como GeoTIFF."""
    count, height, width = array.shape
    with rasterio.open(
        filepath, 'w', driver='GTiff',
        height=height, width=width, count=count,
        dtype=dtype, crs=crs, transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(array.astype(dtype))


def ler_bbox_do_vetorial(caminho_vetor):
    """
    Lê um arquivo vetorial (GPKG) e retorna a bounding box
    no formato (minx, miny, maxx, maxy) em graus decimais (EPSG:4326).
    """
    print(f"Lendo vetor: {caminho_vetor}")
    gdf = gpd.read_file(caminho_vetor)

    if gdf.crs is None:
        print("  [!] AVISO: O vetor não tem CRS definido. Assumindo EPSG:4326.")
    elif gdf.crs.to_epsg() != 4326:
        print(f"  [~] Reprojeteando de {gdf.crs} para EPSG:4326...")
        gdf = gdf.to_crs("EPSG:4326")

    bbox = gdf.total_bounds  # [minx, miny, maxx, maxy]
    print(f"  Bounding box (EPSG:4326): {bbox}")
    return bbox


def calcular_centro_e_edge(bbox):
    """
    A partir de uma bbox (minx, miny, maxx, maxy) em graus,
    calcula o ponto central e o edge_size (em pixels) necessário
    para cobrir a área na resolução desejada.
    """
    minx, miny, maxx, maxy = bbox
    centro_lon = (minx + maxx) / 2.0
    centro_lat = (miny + maxy) / 2.0

    # largura e altura em graus
    largura_graus = maxx - minx
    altura_graus  = maxy - miny

    # converte para metros aproximados (1 grau ~ 111320m no equador)
    # Para latitude, usamos cos(centro_lat) para ajuste
    from math import cos, radians
    fator_lat = cos(radians(centro_lat))
    largura_m = largura_graus * 111320 * fator_lat
    altura_m  = altura_graus * 111320

    # edge_size em pixels na resolução desejada (arredondado para cima)
    edge_largura = int(np.ceil(largura_m / RESOLUCAO_M))
    edge_altura  = int(np.ceil(altura_m / RESOLUCAO_M))
    edge_size = max(edge_largura, edge_altura)

    # O modelo SEN2SR espera que edge_size * FATOR_SUPER_RESOLUCAO (4×) seja
    # um valor suportado internamente (a máscara low-pass tem tamanho fixo).
    # Forçamos edge_size = 128 para garantir compatibilidade com o modelo.
    # (Isso cobre ~1.28 km × 1.28 km centrado no ponto de interesse.)
    edge_size = 128

    print(f"  Centro: ({centro_lat:.6f}, {centro_lon:.6f})")
    print(f"  Dimensão em graus: {largura_graus:.6f} x {altura_graus:.6f}")
    print(f"  Dimensão em metros (aprox): {largura_m:.0f} x {altura_m:.0f}")
    print(f"  Edge size usado: {edge_size} px (fixo para compatibilidade com o modelo)")

    return centro_lat, centro_lon, edge_size


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    # 0. Pré-verificação de DNS
    print("=" * 60)
    print("SEN2SR - Super-resolução Sentinel-2")
    print("=" * 60)

    print("\nVerificando conectividade com servidores Sentinel-2...")
    if not check_dns_resolution(BLOB_HOST):
        print(f"  [!] DNS NÃO RESOLVE: {BLOB_HOST}")
        if not check_internet_connectivity():
            print("  [!!] SEM CONEXÃO COM A INTERNET. Verifique sua rede.\n")
        else:
            print("  [~] Internet OK, mas host específico do blob não resolve.\n")
    else:
        print(f"  [OK] DNS resolvido: {BLOB_HOST}\n")

    # 0.5 Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    # 1. Ler o vetor e extrair a bounding box
    bbox = ler_bbox_do_vetorial(VETOR_LIMITE)
    centro_lat, centro_lon, edge_size_calc = calcular_centro_e_edge(bbox)

    # 2. Seleção do modelo
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
    print(f"Baixando/verificando pesos do modelo em {model_dir}...")
    mlstac.download(file=model_url, output_dir=model_dir)

    # 3. Cubo Sentinel-2 a partir do vetor
    print(f"\nCriando cubo Sentinel-2 ({', '.join(BANDAS)})...")
    print(f"  Período: {START_DATE} a {END_DATE}")
    print(f"  Resolução: {RESOLUCAO_M}m")
    print(f"  Edge size: {edge_size_calc} px")

    da = cubo.create(
        lat=centro_lat,
        lon=centro_lon,
        collection=COLECAO,
        bands=BANDAS,
        start_date=START_DATE,
        end_date=END_DATE,
        edge_size=edge_size_calc,
        resolution=RESOLUCAO_M,
    )

    # CRS e transform
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

    print(f"  Shape original:  {original_s2_numpy.shape}")
    print(f"  Shape super:     {super_resolved_numpy.shape}")

    # 6. Salvamento
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Original 10m
    original_path = os.path.join(OUTPUT_DIR, "original_10m.tif")
    save_geotiff(original_s2_numpy, original_transform, original_crs, original_path)
    print(f"Original salvo em: {original_path}")

    # Super-resolvido 2.5m (transformação ajustada pelo fator)
    new_transform = rasterio.Affine(
        original_transform.a / FATOR_SUPER_RESOLUCAO,
        original_transform.b,
        original_transform.c,
        original_transform.d,
        original_transform.e / FATOR_SUPER_RESOLUCAO,
        original_transform.f
    )
    super_path = os.path.join(OUTPUT_DIR, "super_resolved_2_5m.tif")
    save_geotiff(super_resolved_numpy, new_transform, original_crs, super_path)
    print(f"Super-resolvido salvo em: {super_path}")

    # 7. Visualização
    print("Gerando comparação visual...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(X[[0, 1, 2]].permute(1, 2, 0).cpu().numpy() * 1.5)
    axes[0].set_title("Sentinel-2 Original (10m)")
    axes[0].axis('off')
    axes[1].imshow(superX[[0, 1, 2]].permute(1, 2, 0).cpu().numpy() * 1.5)
    axes[1].set_title("SEN2SR Super-resolved (2.5m)")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    print("\n[OK] Processamento concluído!")


if __name__ == "__main__":
    main()