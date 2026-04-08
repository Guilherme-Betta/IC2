import os
import pandas as pd
from datetime import datetime
from pathlib import Path

# Para utilizar caminhos relativos nos jupyter notebooks
if 'jupyter_notebooks' in os.getcwd():  # Checa se o arquivo é um jupyter notebook
    os.chdir('..')                      # Move o arquivo root para um nível acima do jupyter notebook, o que deveria
                                        # permitir utilizar caminhos relativos ao invés de absolutos

# Constantes
METADADOS = ['Data', 'data_normalizada', 'Estatistica']

# Leitura de arquivos
def dados(caminho_arquivo):
    """
    Carrega dados de arquivos Excel, e separa os metadados das variáveis.
    caminho_arquivo: Caminho relativo do arquivo no workspace.
    """

    # Verifica a existência do arquivo desejado
    try:
        df = pd.read_excel(caminho_arquivo)
    # Retorna uma mensagem caso o arquivo não seja encontrado
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
    
    # Separa os metadados presentes no dataframe das variaveis
    metadados = [col for col in METADADOS if col in df.columns]
    variaveis = [col for col in df.columns if col not in metadados]
    
    return df, metadados, variaveis

# Pivot de dataframes
def pivot(df, metadados, variaveis):
    """
    Pivot do dataframe, criando 3 colunas para cada variável, uma para cada estatística, 
    assim eliminando a coluna "Estatistica" do dataframe
    """
    # Cria listas locais para separar metadados e variaveis do dataframe
    metadados = [col for col in METADADOS if col in df.columns]
    variaveis = [col for col in df.columns if col not in metadados]
    
    # Separa a coluna de Estatisticas para o pivot
    index_cols = [col for col in metadados if col != 'Estatistica']
    
    # Ordena as estatísticas em ordem crescente
    df['Estatistica'] = pd.Categorical(
        df['Estatistica'], 
        categories=['Min.', 'Med.', 'Max.'], 
        ordered=True
    )

    # Executa o pivot
    df_pivot = df.pivot(index=index_cols, 
                        columns='Estatistica',
                        values=variaveis)
    
    # Renomeia as colunas das variáveis, implementando qual estatistica cada coluna representa
    df_pivot.columns = [f"{var}_{est}" for var, est in df_pivot.columns]
    
    # Retorna o dataframe
    return df_pivot.reset_index()

#Unpivot de dataframes
def unpivot(df):
    """
    Unpivot de dataframe.
    """
    # A fazer
    pass

# Salvarmento de arquivos
def salvar(arquivo, nome_arquivo):
    """
    Salva o arquivo em data/out (no workspace). O nome do arquivo é selecionado
    com "nome_arquivo", e um timestamp é automaticamente é adicionado de acordo 
    com o momento que o arquivo for salvo.
    """
    # Cria uma variável para guardar o caminho de onde o arquivo resultante será salvo
    pasta_saida = Path("data/out")

    # Cria um diretorio para salvar o arquivo caso ele ainda nao exista  
    pasta_saida.mkdir(parents=True, exist_ok=True)  

    # Gera o nome do arquivo de saida com timestamp de quando foi criado
    nome_arquivo = str(nome_arquivo + f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

    # Cria o diretório para salvar o arquivo caso ela ainda não exista
    caminho_saida = pasta_saida / nome_arquivo

    # Gera o arquivo de saida
    arquivo.to_excel(caminho_saida, engine='openpyxl', index=True)

    # Gera uma mensagem para o usuario informando sobre onde o arquivo foi salvo
    print(f"Arquivo salvo em: {caminho_saida}")