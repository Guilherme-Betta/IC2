# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv (3.14.2.final.0)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
from pathlib import Path
from datetime import datetime

# %%
# Lista fixa com metadados que podem estar presentes nas bases de dados

METADADOS = ['Data', 'data_normalizada', 'Estatistica', 'Index']

# %%
# Variável direcionada ao diretório raiz do workspace

BASE_DIR = Path(__file__).resolve().parents[2]


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
    variáveis. 
    
    Se o arquivo for designado como original, a função calcula e armazena
    o número de casas decimais utilizadas em cada variável no dicionário
    n_casas_decimais.

    Parameters
    ----------
    caminho_arquivo : str or Path
        Caminho relativo para o diretório do arquivo.
    original : bool, default False
        Define se o arquivo deve ser tratado como a base de dados original.
        Se True, gera o dicionário de casas decimais.

    Returns
    -------
        df : pd.DataFrame
            DataFrame a ser carregado a partir do arquivo Excel.
        metadados : list
            Lista com os nomes das colunas identificadas como metadados de df.
        variaveis : lists
            Lista com nomes das colunas identificadas como variáveis de df.
        n_casas_decimais : dict
            Dicionário com o número de casas decimais utilizadas por variável na base de dados original, 
            caso original=True. Caso contrário, não é retornado.

    Raises
    ------
    FileNotFoundError
        Caso o arquivo não seja localizado no diretório selecionado em "caminho_arquivo".
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
        from pcj.format import casas_decimais
        n_casas_decimais = {}

        for i in variaveis:
            n_casas_decimais[i] = casas_decimais(df[i])

        from pcj.format import casas_decimais_padrao
        casas_decimais_padrao(n_casas_decimais)

        return df, metadados, variaveis, n_casas_decimais
    
    return df, metadados, variaveis


# %%
def salvar(arquivo, nome_arquivo, index=False):
    """
    Salva um arquivo em data/out (no workspace) com timestamp em seu nome.

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
    Salva figuras em data/out (no workspace), com timestamp em seu nome.

    Parameters
    ----------
    fig : str
        Nome da variável contendo a figura a ser salva.
    nome_arquivo : str
        Nome do arquivo em que a figura será salva.
    formato : str, default "png"
        Define o formato em que a figura deve ser salva.
    dpi : float or 'figure', default: 300
        Resolução em dots per inch. Consultar documentação "matplotlib.pyplot.savefig".

    Returns
    -------
        caminho_saida : str
            Caminho do diretório em que o arquivo foi salvo
    """
    pasta_saida = BASE_DIR / "data" / "output" / "figures"
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
    assim eliminando a coluna "Estatistica" do dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame Wide Format a ser pivotado.
    metadados : list
        Lista de metadados presentes em df.
    variaveis : list
        Lista de variáveis presentes em df.

    Returns
    -------
    df_pivot.reset_index() : pd.DataFrame
        DataFrame pivotado para Long Format.
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
    Unpivot de dataframe, revertendo a operação de pivot().
    
    Converte o DataFrame de formato wide (com colunas como 'var_estatistica') 
    para formato long (com colunas de variáveis e uma coluna 'Estatistica').
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame em formato wide, resultado da função pivot().
    
    Returns
    -------
    df_unpivot : pd.DataFrame
        DataFrame em formato long, com metadados, Estatistica e variáveis,
        ordenado na sequência original (Min. -> Med. -> Max.).
    """
    # Identifica os metadados presentes (exceto 'Estatistica' que será recriada)
    metadados_presentes = [col for col in METADADOS if col in df.columns and col != 'Estatistica']
    
    # Identifica as colunas de dados pivotadas (aquelas que não são metadados)
    valor_cols = [col for col in df.columns if col not in metadados_presentes]
    
    # Faz melt do dataframe: converte de wide para long
    df_melted = df.melt(id_vars=metadados_presentes, 
                        value_vars=valor_cols,
                        var_name='var_stat', 
                        value_name='valor')
    
    # Separa a coluna 'var_stat' em variável e estatística
    # Usa rsplit com n=1 para pegar apenas o último "_" (em caso de variáveis com "_" no nome)
    df_melted[['variavel', 'Estatistica']] = df_melted['var_stat'].str.rsplit('_', n=1, expand=True)
    
    # Remove a coluna temporária
    df_melted = df_melted.drop('var_stat', axis=1)
    
    # Faz pivot para voltar às variáveis como colunas
    df_unpivot = df_melted.pivot(index=metadados_presentes + ['Estatistica'],
                                  columns='variavel',
                                  values='valor')
    
    # Reseta o índice
    df_unpivot = df_unpivot.reset_index()
    
    # Remove o nome da coluna de índice
    df_unpivot.columns.name = None
    
    # Ordena as estatísticas na sequência correta (Min. -> Med. -> Max.)
    df_unpivot['Estatistica'] = pd.Categorical(
        df_unpivot['Estatistica'], 
        categories=['Min.', 'Med.', 'Max.'], 
        ordered=True
    )
    
    # Ordena o dataframe pela ordem de Estatistica
    df_unpivot = df_unpivot.sort_values(by=metadados_presentes + ['Estatistica'])
    
    # Reseta o índice para remover os índices desordenados
    df_unpivot = df_unpivot.reset_index(drop=True)
    
    # Converte Estatistica de volta para string (para ficar igual ao original)
    df_unpivot['Estatistica'] = df_unpivot['Estatistica'].astype(str)
    
    return df_unpivot


# %%
def remover_nan(df, salvar_arquivo=False, nome_saida=None):
    """
    Remove linhas com NaN do DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse, contendo NaN.
    salvar_arquivo, default False
        Se True, salva um Excel contendo o DataFrame sem NaN.
    nome_saida : str, optional
        Nome para o arquivo salvo.
        Exemplo: nome_saida="dados" --> "dados_no_nan"

    Returns
    -------
    df : pd.DataFrame
        DataFrame, pós remoção de linhas com NaN.
    """    
    df = df.dropna().copy()

    if salvar_arquivo:
        salvar(df, f"{nome_saida}_no_nan")

    return df
