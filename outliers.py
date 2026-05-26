# %%
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from utils import dados, pivot, salvar

# %% [markdown]
# # Detecção de Outliers

# %% [markdown]
# Métodos estatísticos:
# - Z-Score;
# - IQR;
# - Z-Test - Para ser útil, precisa-se de valores médios para cada variável para colocar no argumento "value=" do método utilizado.
# 
# Métodos de Regressão:
# - NAIVE;
# - ARIMA.
# 
# Métodos Multivariaados Não Supervisionados:
# - LOF;
# - STRAY;
# - Distance From The Mean.
# 
# Clustering:
# - PCA --> K-Means.

# %% [markdown]
# #### Z-Score

# %%
def detect_outliers_zscore(df, variaveis, threshold=3):
    """
    Gera um quadro de outliers de um DataFrame,
    detectados pelo método Z-Score.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list
        Lista de variáveis presentes em df.
    threshold : float, default 3
        Valor de Z-Score máximo para um dado não ser considerado um outlier. 

    Returns
    -------
    outliers_por_coluna_z : np.ndarray
        Número de outliers detectados em cada variável.
    total_outliers_z : int
        Número de linhas com pelo menos um outlier.
    outliers_zscore : np.ndarray of bool
        Máscara booleana indicando se cada valor foi classificado como outlier.
    """    
    # Calcular Z-Scores para todas as colunas numéricas
    z_scores = zscore(df[variaveis])
    
    # Identificar outliers: |z| > threshold (3 é um threshold comum)
    outliers_zscore = (np.abs(z_scores) > threshold)
    
    # Contar outliers por coluna
    outliers_por_coluna_z = outliers_zscore.sum(axis=0)
    
    # Total de linhas com pelo menos um outlier
    total_outliers_z = outliers_zscore.any(axis=1).sum()
    
    return outliers_por_coluna_z, total_outliers_z, outliers_zscore

# %% [markdown]
# ##### IQR

# %%
def detect_outliers_iqr(df, variaveis):
    """
    Gera um quadro de outliers de um DataFrame,
    detectados pelo método IQR.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list
        Lista de variáveis presentes em df.

    Returns
    -------
    total_outliers_iqr : int
        Número total de outliers (somando todas as variáveis).
    total_linhas_outliers_iqr : int
        Número total de linhas com ao menos um outlier.
    outliers_por_linha_iqr : np.ndarray of bool
        Máscara booleana por linha (``True`` para linhas com ao menos um outlier).
    """   
    print("Outliers detectados por IQR (Tukey):")
    
    # Contagem total de outliers
    total_outliers_iqr = 0

    # Contagem de outliers por linha
    outliers_por_linha_iqr = np.zeros(len(df), dtype=bool)
    
    for coluna in variaveis:
        Q1 = df[coluna].quantile(0.25)
        Q3 = df[coluna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        outliers = (df[coluna] < limite_inferior) | (df[coluna] > limite_superior)
        count = outliers.sum()
        print(f"{coluna}: {count} outliers (limites: {limite_inferior:.3f} a {limite_superior:.3f})")
        total_outliers_iqr += count
        outliers_por_linha_iqr |= outliers
    
    total_linhas_outliers_iqr = outliers_por_linha_iqr.sum()
    print(f"\nTotal de outliers (IQR, somando todas as colunas): {total_outliers_iqr}")
    print(f"Total de linhas com pelo menos um outlier (IQR): {total_linhas_outliers_iqr}")
    
    return total_outliers_iqr, total_linhas_outliers_iqr, outliers_por_linha_iqr

# %% [markdown]
# ##### LOF

# %%
def detect_outliers_lof(df, variaveis, n_neighbors=20, contamination='auto'):
    """
    Gera um quadro de outliers de um DataFrame,
    detectados pelo método LOF.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list
        Lista de variáveis presentes em df.
    n_neighbors : int, default 20
        Determina o número de vizinhos a serem considerados pelo algoritmo
        "LocalOutlierFactor" da biblioteca sklearn.
    contamination : string, default 'auto'
        Determina o método de contaminação a ser considerado pelo algoritmo
        "LocalOutlierFactor" da biblioteca sklearn.

    Returns
    -------
    num_outliers : int
        Armazena o número de outliers em df.
    outlier_labels : np.ndarray of int
        Rótulos previstos pelo LOF (``1`` para normal, ``-1`` para outlier).
    outlier_scores : np.ndarray of float
        Scores LOF calculados para cada linha de ``df``.
    """    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outlier_labels = lof.fit_predict(df[variaveis])
    outlier_scores = lof.negative_outlier_factor_
    
    num_outliers = (outlier_labels == -1).sum()
    print(f"Número de outliers detectados pelo LOF: {num_outliers}")
    print(f"Score LOF mínimo: {outlier_scores.min():.3f}")
    print(f"Score LOF máximo: {outlier_scores.max():.3f}")
    print(f"Média dos scores LOF: {outlier_scores.mean():.3f}")
    
    return num_outliers, outlier_labels, outlier_scores

# %% [markdown]
# ##### PCA --> K-Means

# %%
def detect_outliers_pca_kmeans(df, variaveis, n_clusters=3, random_state=42):
    """
    Gera um quadro de outliers de um DataFrame,
    usando PCA + K-Means.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list
        Lista de variáveis presentes em df.
    n_clusters : int, default 3
        Determina o número de clusters a serem utilizados pelo algoritmo 
        "KMeans" da biblioteca sklearn.
    random_state : int, default 42
        Determina o random_state a ser utilizado pelo algoritmo 
        "KMeans" da biblioteca sklearn.

    Returns
    -------
    num_outliers : int
        Armazena o número de outliers identificados em df.
    limiar_outlier : float
        Armazena o valor de distância máximo para que
        um dado seja considerado um outlier.
    distancias : np.ndarray of float
        Armazena os valores de distâncias.
    """    
    X = df[variaveis].copy()
    X_nonan = X.dropna(axis=0, how='any')
    idx_nonan = X_nonan.index
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_nonan)
    
    pca = PCA(n_components=0.95, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X_pca)
    distancias = np.linalg.norm(X_pca - kmeans.cluster_centers_[cluster_labels], axis=1)
    
    limiar_outlier = np.percentile(distancias, 95)
    num_outliers = (distancias > limiar_outlier).sum()
    
    print('K-Means aplicado em', len(idx_nonan), 'pontos. Clusters:', np.unique(cluster_labels))
    print('Número de outliers K-Means (dist > 95%):', num_outliers)
    print('Limiar de distância:', limiar_outlier)
    
    return num_outliers, limiar_outlier, distancias

# %% [markdown]
# ##### Todos

# %%
def detect_outliers(df, variaveis, methods=None):
    """
    Executa múltiplos métodos de detecção de outliers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list
        Lista de variáveis presentes em df.
    methods : list of str, default None
        Métodos de detecção a executar. Se ``None``, todos os métodos
        disponíveis (Z-Score, IQR, LOF, PCA + K-Means) são executados.

    Returns
    -------
    results : dict
        Armazena os resultados do(s) método(s) de detecção utilizados.
    """    
    if methods is None:
        methods = ['zscore', 'iqr', 'lof', 'pca_kmeans']
    
    results = {}
    
    if 'zscore' in methods:
        results['zscore'] = detect_outliers_zscore(df, variaveis)
    
    if 'iqr' in methods:
        results['iqr'] = detect_outliers_iqr(df, variaveis)
    
    if 'lof' in methods:
        results['lof'] = detect_outliers_lof(df, variaveis)
    
    if 'pca_kmeans' in methods:
        results['pca_kmeans'] = detect_outliers_pca_kmeans(df, variaveis)
    
    return results

# %% [markdown]
# # Remoção de Outliers

# %% [markdown]
# ##### Z-Score

# %%
def remove_outliers_zscore(df, variaveis, threshold=3, salvar_arquivo=True):
    """
    Remove outliers de acordo com o método Z-Score.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list
        Lista de variáveis presentes em df.
    threshold : float, default 3
        Valor de Z-Score máximo para um dado não ser considerado um outlier.
    salvar_arquivo : bool, default True
        Determina se um arquivo Excel do DataFrame com os outliers removidos deve ser salvo.

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame com outliers removidos.
    """    
    _, _, outliers_zscore = detect_outliers_zscore(df, variaveis, threshold)
    keep = ~outliers_zscore.any(axis=1)
    df_clean = df.loc[keep].copy()
    
    if salvar_arquivo:
        salvar(df_clean, "outliers_removed_zscore")
    
    return df_clean

# %% [markdown]
# ##### IQR

# %%
def remove_outliers_iqr(df, variaveis, salvar_arquivo=True):
    """
    Remove outliers de acordo com o método IQR (Interquartile Range).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list
        Lista de variáveis presentes em df.
    salvar_arquivo : bool, default True
        Determina se um arquivo Excel do DataFrame com os outliers removidos deve ser salvo.

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame com outliers removidos.
    """    
    _, _, outliers_por_linha_iqr = detect_outliers_iqr(df, variaveis)
    df_clean = df.loc[~outliers_por_linha_iqr].copy()
    
    if salvar_arquivo:
        salvar(df_clean, "outliers_removed_iqr")
    
    return df_clean

# %% [markdown]
# ##### LOF

# %%
def remove_outliers_lof(df, variaveis, n_neighbors=20, contamination='auto', salvar_arquivo=True):
    """
    Remove outliers de acordo com o método LOF (Local Outlier Factor).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list
        Lista de variáveis presentes em df.
    n_neighbors : int, default 20
        Determina o número de vizinhos a serem considerados pelo algoritmo
        "LocalOutlierFactor" da biblioteca sklearn.
    contamination : string, default 'auto'
        Determina o método de contaminação a ser considerado pelo algoritmo
        "LocalOutlierFactor" da biblioteca sklearn.
    salvar_arquivo : bool, default True
        Determina se um arquivo Excel do DataFrame com os outliers removidos deve ser salvo.

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame com outliers removidos.
    """    
    _, outlier_labels, _ = detect_outliers_lof(df, variaveis, n_neighbors, contamination)
    df_clean = df.loc[outlier_labels == 1].copy()
    
    if salvar_arquivo:
        salvar(df_clean, "outliers_removed_lof")
    
    return df_clean

# %% [markdown]
# ##### Todos

# %%
def remove_outliers(df, variaveis, method='zscore', salvar_arquivo=True, **kwargs):
    """
    Remove outliers com o método especificado.
    Usa as funções previamente definidas para
    realizar a remoção.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list
        Lista de variáveis presentes em df.
    method : str, default 'zscore'
        Determina o método de remoção ('zscore', 'iqr', 'lof')
        de outliers a ser utilizado.
    salvar_arquivo : bool, default True
        Determina se um arquivo Excel do DataFrame com os outliers removidos deve ser salvo.
    **kwargs : dict
        Parâmetros adicionais repassados para o método escolhido
        (threshold, n_neighbors, etc.).

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame com outliers removidos pelo método selecionado.

    Raises
    ------
    ValueError
        Caso o método fornecido não esteja na lista ('zscore', 'iqr', 'lof'),
        retorna `f"Unknown method: {method}"`.
    """    
    if method == 'zscore':
        return remove_outliers_zscore(df, variaveis, salvar_arquivo=salvar_arquivo, **kwargs)
    elif method == 'iqr':
        return remove_outliers_iqr(df, variaveis, salvar_arquivo=salvar_arquivo)
    elif method == 'lof':
        return remove_outliers_lof(df, variaveis, salvar_arquivo=salvar_arquivo, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


