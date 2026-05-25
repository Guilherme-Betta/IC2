# %%
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.impute import KNNImputer

import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from utils import dados, pivot, salvar

# %%
def filtro_colunas(df, min_missing, max_missing):
    """
    Seleciona colunas com dados faltantes entre min_missing e max_missing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse.
    min_missing : float
        Determina a fração mínima de dados faltantes tolerados numa coluna.
    max_missing : float
        Determina a fração máxima de dados faltantes tolerados numa coluna.

    Returns
    -------
    colunas_filtradas : list
        Lista com os nomes das colunas com dados faltantes entre min_missing e max_missing.
    """    
    from utils import separar_colunas
    metadados, variaveis = separar_colunas(df)

    # Calcula a porcentagem de dados faltantes para cada coluna
    fracao_faltantes = df[variaveis].isna().mean()

    # Filtra colunas que estão no intervalo [min_missing, max_missing]
    return fracao_faltantes[(fracao_faltantes >= min_missing) & (fracao_faltantes <= max_missing)].index.tolist()

# %%
# Teste: filtra colunas com dados faltantes entre 5% e 15% (para KNN)
df, metadados, variaveis = dados(r"data\input\dados_brutos.xlsx")
filtro_colunas(df, min_missing=0.05, max_missing=0.15)

# %% [markdown]
# # Média

# %% [markdown]
# Imputação para colunas com <5% de dados faltantes

# %%
def mean_imput(df, max_missing=0.05, return_reduced=False, 
                save_missing_pct=False, salvar_arquivo=False):
    """
    Faz a imputação dos dados faltantes via média.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse.
    max_missing : float, default 0.05
        Determina a fração máxima de dados faltantes tolerados numa coluna.
        Se a fração de dados faltantes for maior que max_missing, o algoritmo de 
        imputação não será aplicado à coluna.
    return_reduced : bool, default False
        Se True, retorna apenas as colunas que receberam imputação.
    save_missing_pct : bool, default False
        Se True, retorna um dicionário com a porcentagem de dados faltantes 
        de cada coluna imputada.
    salvar_arquivo : bool, default False
        Determina se um arquivo Excel com o DataFrame pós imputação deve ser gerado.

    Returns
    -------
    df : pd.DataFrame
        DataFrame pós imputação.
    missing_pct : dict, optional
        Dicionário com as porcentagens de dados faltantes (apenas se save_missing_pct=True).
    """
    df = df.copy()

    from utils import separar_colunas
    metadados, variaveis = separar_colunas(df)

    cols = filtro_colunas(df, min_missing=0, max_missing=max_missing)
    
    # Calcula porcentagem de dados faltantes antes de imputar
    if save_missing_pct:
        missing_pct = df[cols].isna().mean().to_dict()
    
    # Faz a imputação
    df[cols] = df[cols].fillna(df[cols].mean())
    
    # Retorna um DataFrame sem as variáveis que não passaram pela imputação
    if return_reduced:
        # Lista os nomes das colunas desejadas no DataFrame final.
        colunas = set(cols + metadados)
        # Busca as colunas presentes na lista anterior.
        colunas = [col for col in df.columns if col in colunas]
        # Reduz o DataFrame às colunas de metadados + variáveis imputadas.
        df = df[colunas]
    
    if salvar_arquivo:
        salvar(df, "imput_mean")
    
    if save_missing_pct:
        return df, missing_pct
    
    return df

# %%
# Teste
dados_limpos, metadados, variaveis = dados(r"data\input\dados_limpos.xlsx")

mean_imput(dados_limpos, return_reduced=True)

# %% [markdown]
# # Mediana

# %% [markdown]
# Imputação para colunas com <5% de dados faltantes

# %%
def median_imput(df, metadados, variaveis, max_missing=0.05, 
                 return_reduced=False, save_missing_pct=False, salvar_arquivo=False):
    """
    Faz a imputação dos dados faltantes via mediana.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse.
    max_missing : float, default 0.05
        Determina a fração máxima de dados faltantes tolerados numa coluna.
        Se a fração de dados faltantes for maior que max_missing, o algoritmo de 
        imputação não será aplicado à coluna.
    return_reduced : bool, default False
        Se True, retorna apenas as colunas que receberam imputação.
    save_missing_pct : bool, default False
        Se True, retorna um dicionário com a porcentagem de dados faltantes 
        de cada coluna imputada.
    salvar_arquivo : bool, default False
        Determina se um arquivo Excel com o DataFrame pós imputação deve ser gerado.

    Returns
    -------
    df : pd.DataFrame
        DataFrame pós imputação.
    missing_pct : dict, optional
        Dicionário com as porcentagens de dados faltantes (apenas se save_missing_pct=True).
    """
    df = df.copy()

    from utils import separar_colunas
    metadados, variaveis = separar_colunas(df)

    cols = filtro_colunas(df, min_missing=0, max_missing=max_missing)
    
    # Calcula porcentagem de dados faltantes antes de imputar
    if save_missing_pct:
        missing_pct = df[cols].isna().mean().to_dict()
    
    # Faz a imputação
    df[cols] = df[cols].fillna(df[cols].median())
    
    # Retorna um DataFrame sem as variáveis que não passaram pela imputação
    if return_reduced:
        # Lista os nomes das colunas desejadas no DataFrame final.
        colunas = set(cols + metadados)
        # Busca as colunas presentes na lista anterior.
        colunas = [col for col in df.columns if col in colunas]
        # Reduz o DataFrame às colunas de metadados + variáveis imputadas.
        df = df[colunas]
    
    if salvar_arquivo:
        salvar(df, "imput_median")
    
    if save_missing_pct:
        return df, missing_pct
    
    return df

# %%
# Teste
dados_limpos, metadados, variaveis = dados(r"data\input\dados_limpos.xlsx")

median_imput(dados_limpos, return_reduced=True)

# %% [markdown]
# # KNN
# 
# 5 - 15% de dados faltantes

# %%
def knn_imput(df, metadados, variaveis, min_missing=0.05, max_missing=0.15, 
              return_reduced=False, save_missing_pct=False, salvar_arquivo=False):
    """
    Faz a imputação dos dados faltantes via KNN (método do sklearn).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse.
    min_missing : float, default 0.05
        Determina a fração mínima de dados faltantes tolerados numa coluna.
    max_missing : float, default 0.15
        Determina a fração máxima de dados faltantes tolerados numa coluna.
    return_reduced : bool, default False
        Se True, retorna apenas as colunas que receberam imputação.
    save_missing_pct : bool, default False
        Se True, retorna um dicionário com a porcentagem de dados faltantes 
        de cada coluna imputada.
    salvar_arquivo : bool, default False
        Determina se um arquivo Excel com o DataFrame pós imputação deve ser gerado.

    Returns
    -------
    df : pd.DataFrame
        DataFrame pós imputação.
    missing_pct : dict, optional
        Dicionário com as porcentagens de dados faltantes (apenas se save_missing_pct=True).
    """
    df = df.copy()

    from utils import separar_colunas
    metadados, variaveis = separar_colunas(df)

    cols = filtro_colunas(df, min_missing=min_missing, max_missing=max_missing)

    # Calcula porcentagem de dados faltantes antes de imputar
    if save_missing_pct:
        missing_pct = df[cols].isna().mean().to_dict()
    
    imputer = KNNImputer(n_neighbors=2, weights="uniform")

    # Garante valores numéricos
    X = df[cols].astype(float)

    df[cols] = imputer.fit_transform(X)

    # Verifica se ainda há valores faltantes nas colunas imputadas
    print("Valores faltantes após imputação KNN:")
    print(df[cols].isna().sum())

    # Retorna um DataFrame sem as variáveis que não passaram pela imputação
    if return_reduced:
        # Lista os nomes das colunas desejadas no DataFrame final.
        colunas = set(cols + metadados)
        # Busca as colunas presentes na lista anterior.
        colunas = [col for col in df.columns if col in colunas]
        # Reduz o DataFrame às colunas de metadados + variáveis imputadas.
        df = df[colunas]
    
    if salvar_arquivo:
        salvar(df, "imput_knn")

    if save_missing_pct:
        return df, missing_pct

    return df

# %%
# Teste
dados_limpos, metadados, variaveis = dados(r"data\input\dados_limpos.xlsx")

knn_imput(dados_limpos, return_reduced=True)

# %% [markdown]
# # KNN - Escalonamento
# 
# 15 - 30% de dados faltantes


