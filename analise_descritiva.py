# %%
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from utils import dados, pivot, salvar
from processamento import prep_brutos
from format import *

# %%
def estatisticas_descritivas(df):
    """
    Gera o quadro de estatísticas descritivas do método "pd.describe" do DataFrame de interesse. 

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe de interesse

    Returns
    -------
    df.describe(include='all') : pd.DataFrame
        Quadro completo de estatísticas descritivas do método "pd.describe" de df.
    """
    return df.describe(include='all')

# %% [markdown]
# # Dados Ausentes

# %%
def calculate_missing_percentage(df1, df2):
    """
    Calcula quantidades e porcentagens de dados faltantes por coluna em df1, adicionando essas informações ao df2.

    Parameters
    ----------
    df1 : pd.DataFrame
        DataFrame de interesse.
    df2 : pd.DataFrame
        DataFrame contendo o quadro de estatísticas do DataFrame de interesse.

    Returns
    -------
    result : dict
        Dicionário com duas chaves:
        - ``'by_column'``: ``pd.Series`` com a porcentagem de dados faltantes por coluna.
        - ``'total'``: ``float`` com a porcentagem total de dados faltantes no DataFrame.
    """
    # Soma a quantidade de dados faltantes por coluna
    missing_by_column = df1.isna().sum()

    # Conta a quantidade de linhas no dataframe
    total_rows = len(df1)

    # Calcula a porcentagem de dados faltantes por coluna, a partir dos 2 passos anteriores
    missing_percentage_by_column = (missing_by_column / total_rows * 100).round(2)

    # Calcula o número total de dados faltantes no dataframe
    missing_total = df1.isna().sum().sum()

    # Calcula o total de "células" presentes na base de dados
    total_elements = total_rows * len(df1.columns) # Talvez trocar 'df1.columns' por 'variaveis'

    # Gera a linha com porcentagem de dados faltantes por coluna
    missing_percentage_total = (missing_total / total_elements * 100).round(2)

    # Gera a linha com porcentagem de dados faltantes por coluna
    df2.loc["Porcentagem de Dados Faltantes"] = missing_percentage_by_column

    # Gera a linha com a contagem de dados faltantes por coluna
    df2.loc["Numero de Dados Faltantes"] = missing_by_column
    return {
        'by_column': missing_percentage_by_column,
        'total': missing_percentage_total
    }

# %% [markdown]
# # Dados não numéricos

# %% [markdown]
# Tratando dados não numéricos como "corrompidos"

# %%
def corrupted_data(df1, df2, variaveis):
    """
    Calcula dados corrompidos (não numéricos) e adiciona ao df2.

    Parameters
    ----------
    df1 : pd.DataFrame
        DataFrame de interesse.
    df2 : pd.DataFrame
        DataFrame contendo o quadro de estatísticas do DataFrame de interesse. 
    variaveis : list
        Lista contendo os nomes das colunas a serem tratadas como variáveis em df1 e df2.

    Returns
    -------
    result : dict
        Dicionário com duas chaves:
        - ``'by_column'``: ``pd.Series`` com a porcentagem de dados corrompidos por coluna.
        - ``'total'``: ``float`` com a porcentagem total de dados corrompidos no DataFrame.
    """
    # Cria um dicionário para a contagem de dados corrompidos
    contagem_corrompidos = {}
    for coluna in variaveis:

        # Busca a contagem de dados faltantes por coluna gerada pela funcao de "calculate_missing_percentage"
        nan_original = df2.loc["Numero de Dados Faltantes", coluna]

        # Remove valores de dtype != float das colunas na lista "variaveis"
        coluna_corrigida = pd.to_numeric(df1[coluna], errors='coerce')

        # Conta os dados ausentes apos a remocao do passo anterior
        nan_update = coluna_corrigida.isna().sum()

        # Subtrai a quantidade de NaN apos a remocao pela quantidade de NaN originais
        corrupted_count = nan_update - nan_original

        # Cria a contagem de dados corrompidos no dicionario
        contagem_corrompidos[coluna] = corrupted_count
    
    # Usa o dicionario de dados corrompidos para contar a quantidade de dados corrompidos por coluna
    corrupted_by_column = pd.Series(contagem_corrompidos)
    total_rows = len(df1)
    corrupted_percentage_by_column = (corrupted_by_column / total_rows * 100).round(2)
    corrupted_total = corrupted_by_column.sum()
    total_elements = total_rows * len(variaveis)
    corrupted_percentage_total = (corrupted_total / total_elements * 100).round(2)
    df2.loc["Porcentagem de Dados Corrompidos"] = corrupted_percentage_by_column
    df2.loc["Numero de Dados Corrompidos"] = corrupted_by_column
    return {
        'by_column': corrupted_percentage_by_column,
        'total': corrupted_percentage_total
    }

# %% [markdown]
# # Função para análise descritiva completa
# Reúne todas as funções e executa todos os passos de uma vez

# %%
def analise_descritiva(df, 
                       metadados = None, 
                       variaveis = None, 
                       n_casas_decimais = None, 
                       salvar_arquivo=True, 
                       transpose=True):
    """
    Executa todas as etapas da análise descritiva, e formata os resultados.
    Executando as funções: (i) "estatisticas_descritivas(df)", 
    (ii) "calculate_missing_percentage(df1, df2)", (iii) "corrupted_data(df1, df2, variaveis)",
    (iv) "formatar_variaveis(df, n_casas_decimais)", (v) "formatar_metadados(df, metadados)"

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse.
    metadados : list, default None
        Lista com os nomes das colunas identificadas como metadados de df.
    variaveis : list, default None
        Lista com nomes das colunas identificadas como variáveis de df.
    n_casas_decimais : dict, default None
        Dicionário contendo o número padrão de casas decimais para cada variável.
    salvar_arquivo : bool, default True
        Determina se um arquivo Excel com o quadro de estatísticas descritivas deve ser gerado.
    transpose : bool, default True
        Determina se o quadro de estatísticas deve ser transposto.

    Returns
    -------
    describe : pd.DataFrame
        Variável contendo o quadro de estatísticas final.
    """

    # Determina as listas de metadados e variaveis com base em df
    if metadados is None or variaveis is None:
        from utils import separar_colunas
        metadados, variaveis = separar_colunas(df)

    # Determina o número de casas decimais a ser utilizada em cada variavel com base no dicionário gerado na leitura dos dados originais
    if n_casas_decimais is None:
        n_casas_decimais = n_casas_decimais

    # Executa as etapas desejadas para o quadro estatístico completo, armazenando-o em "describe"
    describe = estatisticas_descritivas(df)
    calculate_missing_percentage(df, describe)
    corrupted_data(df, describe, variaveis)
    formatar_variaveis(describe, n_casas_decimais)
    describe = formatar_metadados(describe, metadados)
    if transpose:
        describe = describe.T
    if salvar_arquivo:
        salvar(describe, "describe", index=True)

    return describe


