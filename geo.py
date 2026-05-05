# -*- coding: utf-8 -*-
import os
import torch
import mlstac
import cubo
import matplotlib.pyplot as plt

"""
SEN2SR - Super-resolução de imagens Sentinel-2 para 2.5m
Este script aumenta a resolução das bandas de 10m (RGB e NIR) para 2.5m.
"""

# Tenta verificar se o mamba_ssm está disponível para decidir qual modelo usar
try:
    import mamba_ssm
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

# Parâmetros da Área de Interesse (AOI)
LATITUDE = -10.18440098
LONGITUDE = -48.33361440
START_DATE = "2025-09-08"
END_DATE = "2025-09-08"
IMAGE_INDEX = 0

def main():
    # 1. Configuração do dispositivo (GPU é recomendada)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 2. Seleção do modelo e diretório de download
    if HAS_MAMBA:
        model_url = "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SR/NonReference_RGBN_x4/mlm.json"
        model_dir = "model/SEN2SR_RGBN"
        print("Usando arquitetura: Mamba (Full)")
    else:
        model_url = "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json"
        model_dir = "model/SEN2SRLite_RGBN"
        print("Usando arquitetura: SwinIR (Lite)")

    # 3. Download dos pesos do modelo (pula se já existir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print(f"Verificando/Baixando pesos do modelo em {model_dir}...")
    mlstac.download(
        file=model_url,
        output_dir=model_dir,
    )

    # 4. Criação do cubo de dados Sentinel-2 L2A via cubo
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

    # 5. Preparação dos dados (Normalização 1/10000)
    print("Processando imagem original...")
    original_s2_numpy = (da[IMAGE_INDEX].compute().to_numpy() / 10000).astype("float32")
    X = torch.from_numpy(original_s2_numpy).float().to(device)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 6. Carregamento e Execução do Modelo
    print("Carregando e compilando o modelo...")
    model = mlstac.load(model_dir).compiled_model(device=device)

    print("Executando inferência de super-resolução...")
    with torch.no_grad():
        # Adiciona dimensão de batch e remove após processar
        superX = model(X[None]).squeeze(0)

    # 7. Visualização dos Resultados
    print("Gerando comparação visual...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Original (Bandas 0,1,2 = RGB)
    axes[0].imshow(X[[0, 1, 2], :, :].permute(1, 2, 0).cpu().numpy() * 1.5)
    axes[0].set_title("Sentinel-2 Original (10m)")
    axes[0].axis('off')

    # Super-resolvido
    axes[1].imshow(superX[[0, 1, 2], :, :].permute(1, 2, 0).cpu().numpy() * 1.5)
    axes[1].set_title("SEN2SR Super-resolved (2.5m)")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()