import pandas as pd
import os
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from utils import dados, pivot, salvar

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

def casas_decimais(coluna):
    """
    Funcao que determina a quantidade de casas decimais utilizadas em UMA coluna
    de um banco de dados.
    Serah utilizada para padronizar o numero de casas decimais por variavel, com base
    no numero de casas decimais utilizadas no banco original.
    """
    # Elimina celulas vazias e converte todos os valores da coluna para string
    textos = coluna.dropna().astype(str)    

    # Busca celulas que contem "." 
    decimais = textos[textos.str.contains(r'\.', regex=True, na=False)] 

    if decimais.empty:
        return 0    # Colunas sem "." sao definidas como contendo 0 casas decimais
    
    n_decimais = decimais.str.split('.').str[1].str.len()

    # Retorna o valor máximo de dígitos após a casa decimal como integer
    return int(n_decimais.max())

def formatar_variaveis(describe, dados, variaveis):
    """
    Formata o DataFrame describe com casas decimais baseadas nos dados originais.
    describe: DataFrame de estatísticas.
    dados: DataFrame original.
    variaveis: Lista de colunas variáveis.
    """

    # Selecao de estatisticas cujos valores devem ser formatados. 
    # Exclui as estatisticas criadas e adicionadas a tabela,
    # pois essas sao porcentagens ou contagens (numeros inteiros), que serao 
    # padronizadas com 2 e 0 casas decimais respectivamente
    estatisticas_para_formatar = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']

    # Loop para iterar sobre as variaveis numericas
    for col in variaveis:
        
        # 1. A função analisa UMA coluna do banco ORIGINAL por vez
        casas = casas_decimais(dados[col])

        # 2. Criacao da regra de formatacao com base na analise do passo anterior (ex: se casas=2, vira "{:.2f}")
        regra_formatacao = f"{{:.{casas}f}}"

        # 3. Aplicacao da formatacao nas estatísticas selecionadas no primeiro passo
        dados_matematicos = describe.loc[estatisticas_para_formatar, col]
        describe.loc[estatisticas_para_formatar, col] = dados_matematicos.apply(
            lambda x: regra_formatacao.format(float(x)) if pd.notna(x) else x
        )
    return describe

def formatar_metadados(describe, metadados):
    """
    Limpa o DataFrame describe, removendo estatísticas desnecessárias para metadados.
    describe: DataFrame de estatísticas.
    metadados: Lista de colunas metadados.
    estatisticas_manter: Estatísticas a manter para metadados.
    transpose: Permite selecionar se o quadro de estatísticas deve ser pivotado ou não (Pivotado by default).
    """
    # Lista de estatisticas de interesse para as colunas de metadados
    estatisticas_manter = ['count', 'unique', 'freq']

    # Cria uma lista das estatisticas que nao estao na lista do passo anterior
    estatisticas_apagar = [stat for stat in describe.index if stat not in estatisticas_manter]

    # Elimina as estatisticas que nao estao na lista do primeiro passo
    describe.loc[estatisticas_apagar, metadados] = ''

    # Retorna o quadro de estatísticas
    return describe

def analise_descritiva(df, metadados, variaveis, salvar_arquivo=True, transpose=True):
    """
    Executa todas as etapas da análise descritiva
    df: DataFrame cujas estatísticas estão sendo descritas
    metadados: Lista de colunas de metadados
    variaveis: Lista de colunas com variaveis numéricas
    salvar_arquivo: Determina se o quadro será salvo (Ativado by default)
    """
    describe = estatisticas_descritivas(df)
    calculate_missing_percentage(df, describe)
    corrupted_data(df, describe, variaveis)
    formatar_variaveis(describe, df, variaveis)
    describe = formatar_metadados(describe, metadados)
    if transpose:
        describe = describe.T
    if salvar_arquivo:
        salvar(describe, "describe", index=True)
    return describe