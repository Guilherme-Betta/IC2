# %%
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

# %%
# Lista fixa com metadados que podem estar presentes nas bases de dados

METADADOS = ['Data', 'data_normalizada', 'Estatistica', 'Index']

# %%
# Variável que permite utilizar caminhos relativos nos jupyter notebooks

BASE_DIR = Path(__file__).resolve().parent

# %%
def separar_colunas(df):
    """
    Separa colunas de metadados de colunas de variaveis.

    Retorna: metadados, variaveis
    """
    metadados = [col for col in METADADOS if col in df.columns]
    variaveis = [col for col in df.columns if col not in metadados]
    return metadados, variaveis

# %%
def dados(caminho_arquivo, original=False):
    """
    Carrega dados de arquivos Excel, e separa os metadados das variáveis.

    caminho_arquivo: Caminho relativo do arquivo no workspace.
    original: Seleciona se a base de dados deve ser tratada como a original (default: False)

    Retorna: df, metadados, variaveis (, n_casas_decimais, se original=True)
    """
    full_path = BASE_DIR / caminho_arquivo
    # Verifica a existência do arquivo desejado
    try:
        df = pd.read_excel(full_path)
    # Retorna uma mensagem caso o arquivo não seja encontrado
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
    
    # Separa os metadados presentes no dataframe das variaveis
    metadados, variaveis = separar_colunas(df)

    # Armazena o número de casas decimais utilizadas por variável na base de dados. 
    # É útil implementar esse método enquanto a base ainda está alterada, portanto,
    # Ela está sendo colocada nessa função dedicada aos dados brutos.

    if original:
        from format import casas_decimais
        n_casas_decimais = {}

        for i in variaveis:
            n_casas_decimais[i] = casas_decimais(df[i])

        from format import casas_decimais_padrao
        casas_decimais_padrao(n_casas_decimais)

        return df, metadados, variaveis, n_casas_decimais
    
    return df, metadados, variaveis

# %%
def salvar(arquivo, nome_arquivo, index=False):
    """
    Salva o arquivo em data/out (no workspace), com timestamp.

    arquivo: Nome do arquivo que deve ser salvo.
    nome_arquivo: Nomeia o arquivo.
    index: Seleciona se um novo index deve ser criado (default: False)
    """

    # Cria uma variável para guardar o caminho de onde o arquivo resultante será salvo
    pasta_saida = BASE_DIR / "data" / "out"

    # Cria um diretorio para salvar o arquivo caso ele ainda nao exista  
    pasta_saida.mkdir(parents=True, exist_ok=True)  

    # Gera o nome do arquivo de saida com timestamp de quando foi criado
    nome_arquivo = str(nome_arquivo + f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

    # Cria o diretório para salvar o arquivo caso ela ainda não exista
    caminho_saida = pasta_saida / nome_arquivo

    # Gera o arquivo de saida
    arquivo.to_excel(caminho_saida, engine='openpyxl', index=index)

# %%
def salvar_visualizacao(fig, nome_arquivo, formato="png", dpi=300):
    """
    Salva a figura em data/out (no workspace), com timestamp.

    fig: Nome da figura a ser salva.
    nome_arquivo: Nomeia o arquivo da figura.
    formato: Formato de arquivo da figura (default: png).
    dpi: Resolução da imagem em "dots per inch" (default: 300).
    """
    pasta_saida = BASE_DIR / "data" / "out"
    pasta_saida.mkdir(parents=True, exist_ok=True)

    nome_arquivo = f"{nome_arquivo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{formato}"
    caminho_saida = pasta_saida / nome_arquivo

    fig.savefig(caminho_saida, dpi=dpi, bbox_inches="tight")
    print(f"Visualização salva em: {caminho_saida}")
    return caminho_saida


