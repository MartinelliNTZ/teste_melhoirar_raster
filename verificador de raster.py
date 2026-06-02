"""
================================================================================
  SENTINEL-2 L2A - VERIFICADOR DE TIPO DE DADO POS-DOWNLOAD
  Compatível com Windows | Python 3.8+
================================================================================
  Verifica metadados dos GeoTIFFs baixados:

    Tipo uint16  -> precisa dividir por 10000 para obter reflectancia real (0-1)
    Tipo float32 -> ja esta em reflectancia (0-1)

  USO:
    python geo11.py

  CONFIGURACAO:
    Edite PASTA_BASE abaixo para apontar para a pasta com os rasters baixados.
================================================================================
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import rasterio
    import numpy as np
except ImportError:
    print("\n[ERRO] rasterio e necessario. Instale com:")
    print("  pip install rasterio numpy\n")
    sys.exit(1)

# -- CONFIGURACAO --------------------------------------------------------------
PASTA_BASE = r"ourives"
# -----------------------------------------------------------------------------

SEP  = "=" * 80
SEP2 = "-" * 80


def cabecalho():
    print(f"\n{SEP}")
    print("   VERIFICADOR DE TIPO DE DADO - Sentinel-2 L2A")
    print(f"   Pasta: {PASTA_BASE}")
    print(f"{SEP}")


def verificar_arquivo(tif_path: Path) -> dict:
    """Abre um GeoTIFF e extrai metadados de tipo e valores extremos."""
    try:
        with rasterio.open(tif_path) as src:
            dtype   = src.dtypes[0]
            crs     = src.crs
            largura = src.width
            altura  = src.height
            n_bands = src.count
            nodata  = src.nodata

            # Ler amostra dos dados (apenas primeira banda)
            banda = src.read(1)

            # Se tiver nodata, mascarar para estatisticas reais
            if nodata is not None:
                mascara = banda != nodata
                if mascara.any():
                    vmin = float(banda[mascara].min())
                    vmax = float(banda[mascara].max())
                    vmed = float(banda[mascara].mean())
                else:
                    vmin = vmax = vmed = 0.0
            else:
                vmin = float(banda.min())
                vmax = float(banda.max())
                vmed = float(banda.mean())

            return {
                "caminho":  tif_path,
                "nome":     tif_path.name,
                "dtype":    dtype,
                "n_bands":  n_bands,
                "largura":  largura,
                "altura":   altura,
                "nodata":   nodata,
                "crs":      str(crs) if crs else "N/A",
                "min":      vmin,
                "max":      vmax,
                "media":    vmed,
            }
    except Exception as e:
        return {
            "caminho": tif_path,
            "nome":    tif_path.name,
            "dtype":   "ERRO",
            "n_bands": 0,
            "largura": 0,
            "altura":  0,
            "nodata":  None,
            "crs":     "N/A",
            "min":     0,
            "max":     0,
            "media":   0,
            "erro":    str(e),
        }


def status_dtype(dtype: str) -> str:
    """Retorna mensagem sobre o tipo de dado."""
    if dtype == "uint16":
        return "[uint16] precisa dividir por 10000 para reflectancia (0-1)"
    elif dtype == "float32":
        return "[float32] ja esta em reflectancia (0-1)"
    elif dtype.startswith("uint") or dtype.startswith("int"):
        return "[%s] valor inteiro, provavelmente precisa de escala" % dtype
    elif dtype.startswith("float"):
        return "[%s] ponto flutuante, provavelmente ja e reflectancia" % dtype
    else:
        return "[%s] tipo nao classificado" % dtype


def exibir_resultados(arquivos: list, pasta_base: Path):
    """Exibe tabela formatada com resultados da verificacao."""
    total          = len(arquivos)
    uint16_count   = 0
    float32_count  = 0
    outros_count   = 0
    erros_count    = 0

    print(f"\n{SEP2}")
    print("   RESUMO POR ARQUIVO")
    print(f"{SEP2}")
    print("  %-55s %-10s %10s %10s %10s  %s" % ("ARQUIVO", "TIPO", "MIN", "MAX", "MEDIA", "STATUS"))
    print("  %s" % ("-" * 125))

    for info in arquivos:
        nome = info["nome"]
        dtype = info["dtype"]

        if dtype == "ERRO":
            print("  [ERRO] %-55s %-10s %10s %10s %10s  %s" % (nome, "ERRO", "-", "-", "-", info.get("erro", "desconhecido")))
            erros_count += 1
            continue

        # Flag de escala sugerida
        if dtype == "uint16":
            escala = "/10000"
            uint16_count += 1
        elif dtype == "float32":
            escala = "ja OK"
            float32_count += 1
        else:
            escala = "verificar"
            outros_count += 1

        minimo = info["min"]
        maximo = info["max"]
        media  = info["media"]

        # Se uint16 e max > 1000, reforca aviso
        aviso = ""
        if dtype == "uint16" and maximo > 3000:
            aviso = " <<<"

        print("  %-55s %-10s %10.2f %10.2f %10.2f  %s%s" % (nome, dtype, minimo, maximo, media, escala, aviso))

    # -- Resumo final agrupado --------------------------------------------------
    print(f"\n{SEP2}")
    print("   RESUMO GERAL")
    print(f"{SEP2}")

    print(f"\n  Pasta base : {pasta_base.resolve()}")
    print(f"  Total      : {total} arquivo(s)")

    # Por cena (pasta)
    cenas = {}
    for info in arquivos:
        pasta_cena = info["caminho"].parent.name
        if pasta_cena not in cenas:
            cenas[pasta_cena] = {"uint16": 0, "float32": 0, "outros": 0, "erro": 0}
        dt = info["dtype"]
        if dt == "ERRO":
            cenas[pasta_cena]["erro"] += 1
        elif dt == "uint16":
            cenas[pasta_cena]["uint16"] += 1
        elif dt == "float32":
            cenas[pasta_cena]["float32"] += 1
        else:
            cenas[pasta_cena]["outros"] += 1

    print(f"\n  {'CENA':<30} {'uint16':>8} {'float32':>8} {'outros':>8} {'erros':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for cena, cont in sorted(cenas.items()):
        print("  %-30s %8d %8d %8d %8d" % (cena, cont['uint16'], cont['float32'], cont['outros'], cont['erro']))

    # -- Recomendacao final ----------------------------------------------------
    print(f"\n{SEP2}")
    print("   RECOMENDACAO")
    print(f"{SEP2}")

    if uint16_count > 0:
        print(f"\n  >>> ATENCAO: {uint16_count} arquivo(s) sao uint16 com valores 0-12000.")
        print(f"      Para usar como reflectancia real (0-1), divida todos por 10000:")
        print(f"      Exemplo no QGIS: Raster Calculator -> \"B04 / 10000\"")

    if float32_count > 0:
        print(f"\n  >>> OK: {float32_count} arquivo(s) ja estao em float32 (reflectancia 0-1).")
        print(f"      Nenhuma conversao necessaria.")

    if erros_count > 0:
        print(f"\n  >>> ERRO: {erros_count} arquivo(s) com erro de leitura. Verifique integridade dos arquivos.")

    print(f"\n{SEP}\n")


def main():
    cabecalho()

    pasta_base = Path(PASTA_BASE)
    if not pasta_base.exists() or not pasta_base.is_dir():
        print(f"\n  [ERRO] Pasta nao encontrada: {pasta_base.resolve()}")
        sys.exit(1)

    # Coletar todos os .tif recursivamente
    tifs = sorted(pasta_base.rglob("*.tif"))
    # Excluir .aux.xml e arquivos temporarios
    tifs = [t for t in tifs if not t.name.startswith("_tmp") and ".aux.xml" not in t.name]

    if not tifs:
        print(f"\n  [Aviso] Nenhum arquivo .tif encontrado em {pasta_base.resolve()}")
        sys.exit(0)

    print(f"\n  >>> Encontrados {len(tifs)} arquivos .tif para verificar...\n")

    # Verificar cada arquivo
    resultados = []
    for i, tif in enumerate(tifs, 1):
        info = verificar_arquivo(tif)

        # Status inline
        dt = info["dtype"]
        if dt == "ERRO":
            icone = "[ERR]"
        elif dt == "uint16":
            icone = "[U16]"
        elif dt == "float32":
            icone = "[F32]"
        else:
            icone = "[OTH]"

        # Mostrar progresso inline
        nome_curto = info["nome"][:55]
        msg = status_dtype(dt).split("]")[-1].strip() if dt not in ("ERRO",) else "ERRO"
        print("    [%3d/%d] %s %-55s -> %s" % (i, len(tifs), icone, nome_curto, msg))

        resultados.append(info)

    exibir_resultados(resultados, pasta_base)


if __name__ == "__main__":
    main()