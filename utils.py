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
    """Separa colunas em listas de metadados e variáveis presentes no df.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse.

    Returns
    -------
    metadados : list
        Lista de metadados presentes em df.

    variaveis : list
        Lista de variáveis presentes em df.
    """
    metadados = [col for col in METADADOS if col in df.columns]
    variaveis = [col for col in df.columns if col not in metadados]
    return metadados, variaveis

# %%
def dados(caminho_arquivo, original=False):
    """
    Carrega um arquivo, separa as suas colunas em metadados e 
    variáveis. Se o arquivo for designado como original, armazena
    o número de casas decimais utilizadas em cada variável no dicionário
    n_casas_decimais.

    Parameters
    ----------
    caminho_arquivo : str
        Caminho relativo para o diretório do arquivo.
    original : bool, default False
        Define se o arquivo deve ser tratado como a base de dados original.

    Returns
    -------
        df : pd.DataFrame
            DataFrame a ser carregado.
        metadados : list
            Lista com os nomes das colunas identificadas como metadados de df.
        variaveis : lists
            Lista com nomes das colunas identificadas como variáveis de df
        n_casas_decimais : dict
            Dicionário com o número de casas decimais utilizadas por variável na base de dados original, 
            caso original=True. Caso contrário, não é retornado.
        

    Raises
    ------
    FileNotFoundError
        Caso o arquivo não seja localizado no diretório selecionado em "caminho_arquivo"
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
    Salva um arquivo em data/output (no workspace) com timestamp em seu nome.

    Parameters
    ----------
    arquivo : str
        Nome da variável com o conteúdo a ser salvo.
    nome_arquivo : str
        Nome a ser dado ao arquivo no salvamento.
    index : bool, default False
        Define se uma coluna de índice deve ser adicionada para enumerar as linhas do arquivo.
    """

    # Cria uma variável para guardar o caminho de onde o arquivo resultante será salvo
    pasta_saida = BASE_DIR / "data" / "output"

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
    Salva figuras em data/output (no workspace), com timestamp em seu nome.

    Parameters
    ----------
    fig : str
        Nome da variável contendo a figura a ser salva.
    nome_arquivo : str
        Nome do arquivo em que a figura será salva.
    formato : str, default "png"
        Define o formato em que a figura deve ser salva.
    dpi : float or 'figure', default: 300
        Resolução em dots per inch. Consultar documentação "matplotlib.pyplot.savefig"

    Returns
    -------
        caminho_saida : str
            Caminho do diretório em que o arquivo foi salvo
    """
    pasta_saida = BASE_DIR / "data" / "output"
    pasta_saida.mkdir(parents=True, exist_ok=True)

    nome_arquivo = f"{nome_arquivo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{formato}"
    caminho_saida = pasta_saida / nome_arquivo

    fig.savefig(caminho_saida, dpi=dpi, bbox_inches="tight")
    print(f"Visualização salva em: {caminho_saida}")
    return caminho_saida

# %%
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

# %%
def unpivot(df):
    """
    Unpivot de dataframe.
    """
    # A fazer


