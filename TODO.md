# SEN2SR - Super-resolução Sentinel-2

## Implementado em `geo3.py`
- [x] Leitura de vetor GPKG como AOI (polígono/multipolígono)
- [x] Bounding box completa coberta por grade de tiles 128×128
- [x] Pipeline de tiling: fatia o cubo grande → processa cada tile → remonta mosaico
- [x] CRS automático: reprojeção do polígono para UTM (EPSG:32722 para região)
- [x] Máscara do polígono recorta exatamente a forma desejada
- [x] Configurações centralizadas no início do script
- [x] Pré-verificação de DNS e retry com backoff exponencial
- [x] Geração de GeoTIFFs georreferenciados (original + super-resolvido)
- [x] Visualização comparativa salva como PNG

## Último teste (06/05/2026)
- Polígono: ~8.7 km × 5.9 km
- Cubo: 896×896 px (7×7 tiles de 128px)
- 49 tiles processados em GPU (CUDA)
- Saída: `resultados/original_10m_mosaic.tif` + `super_resolved_2_5m_mosaic.tif`
- Área poligonal mascarada: ~50.4 km²