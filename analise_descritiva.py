import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from utils import dados, pivot, salvar
from format import *

def estatisticas_descritivas(df):
    """
    Gera o quadro de estatísticas descritivas do método "describe" do DataFrame de interesse.
    df: DataFrame de interesse.
    """
    return df.describe(include='all')

def calculate_missing_percentage(df1, df2):
    """
    Calcula quantidades e porcentagens de dados faltantes por coluna, adicionando ao df2.
    df1: DataFrame original.
    df2: Quadro de estatísticas (describe) a ser modificado.
    """
    # Soma a quantidade de dados faltantes por coluna
    missing_by_column = df1.isna().sum()

    # Conta a quantidade de linhas no dataframe
    total_rows = len(df1)

    # Calcula a porcentagem de dados faltantes por coluna, a partir dos 2 passos anteriores
    missing_percentage_by_column = (missing_by_column / total_rows * 100).round(2)

    # Calcula o número total de dados faltantes no dataframe
    missing_total = df1.isna().sum().sum()

    # Calcula o total de "células" presentes no banco de dados
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

def corrupted_data(df1, df2, variaveis):
    """
    Calcula dados corrompidos (não numéricos) e adiciona ao df2.
    df1: DataFrame original.
    df2: DataFrame de estatísticas.
    variaveis: Lista de colunas variáveis.
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

def analise_descritiva(df, 
                       metadados = None, 
                       variaveis = None, 
                       n_casas_decimais = None, 
                       salvar_arquivo=True, 
                       transpose=True):
    """
    Executa todas as etapas da análise descritiva
    df: DataFrame cujas estatísticas estão sendo descritas
    metadados: Lista de colunas de metadados
    variaveis: Lista de colunas com variaveis numéricas
    salvar_arquivo: Determina se o quadro será salvo (Ativado by default)
    """

    if metadados is None or variaveis is None:
        from utils import separar_colunas
        metadados, variaveis = separar_colunas(df)

    if n_casas_decimais is None:
        n_casas_decimais = get_casas_decimais_padrao()

    # Filtrar variáveis presentes no dataframe atual
    if n_casas_decimais is not None:
        n_casas_decimais = {k: v for k, v in n_casas_decimais.items() if k in variaveis}

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