import asf_search as asf

# 1. Configure seus dados de acesso (obrigatório)
# Você pode definir as variáveis de ambiente ED_USERNAME e ED_PASSWORD
# ou passar o arquivo de netrc. Vamos usar a autenticação interativa.
asf.authenticate()  # Solicitará seu usuário e senha do Earthdata

# 2. Defina os parâmetros de busca
opts = {
    'platform': 'ALOS',          # Plataforma
    'instrument': 'PALSAR',      # Instrumento (essencial para ALOS)[reference:2]
    'processingLevel': 'TERRAIN',# Produto "Hi-Res Terrain Corrected"
    'beamMode': ['FBS', 'FBD'],  # Modos de feixe comuns para DEM
    'start': '2006-01-01T00:00:00Z', # Data de início (opcional)
    'end': '2011-05-12T00:00:00Z',   # Data de fim (opcional)
    # Defina a área de interesse (WKT)
    'intersectsWith': 'POLYGON((-47.9 -15.8,-47.8 -15.8,-47.8 -15.9,-47.9 -15.9,-47.9 -15.8))'
}

# 3. Realize a busca
print("Buscando cenas...")
results = asf.search(**opts)
print(f"Encontradas {len(results)} cenas.")

# 4. Faça o download
if results:
    print("Iniciando download...")
    results.download(path='./dados_alos/')
    print("Download concluído!")
else:
    print("Nenhuma cena encontrada para os parâmetros fornecidos.")