import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from utils import salvar

def detect_outliers_zscore(df, variaveis, threshold=3):
    """
    Detecta outliers usando Z-Score, baseado no código do notebook.
    
    df: DataFrame.
    variaveis: Lista de colunas numéricas.
    threshold: Limite para |z| (default 3).
    
    Retorna: (outliers_por_coluna, total_outliers_z, outliers_zscore)
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

def detect_outliers_iqr(df, variaveis):
    """
    Detecta outliers usando IQR (Tukey), baseado no código do notebook.
    
    df: DataFrame.
    variaveis: Lista de colunas numéricas.
    
    Retorna: (total_outliers_iqr, total_linhas_outliers_iqr, outliers_por_linha_iqr)
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

def detect_outliers_lof(df, variaveis, n_neighbors=20, contamination='auto'):
    """
    Detecta outliers usando LOF, baseado no código do notebook.
    
    df: DataFrame.
    variaveis: Lista de colunas numéricas.
    
    Retorna: (num_outliers, outlier_labels, outlier_scores)
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

def detect_outliers_pca_kmeans(df, variaveis, n_clusters=3, random_state=42):
    """
    Detecta outliers usando PCA + K-Means, baseado no código do notebook.
    
    df: DataFrame.
    variaveis: Lista de colunas numéricas.
    
    Retorna: (num_outliers, limiar_outlier, distancias)
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

def detect_outliers(df, variaveis, methods=None):
    """
    Executa múltiplos métodos de detecção de outliers.
    
    df: DataFrame.
    variaveis: Lista de colunas numéricas.
    methods: 'zscore', 'iqr', 'lof', 'pca_kmeans' (default: todos).
    
    Retorna: Dicionário com resultados de cada método.
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

def remove_outliers_zscore(df, variaveis, threshold=3, salvar_arquivo=True):
    """
    Remove outliers de acordo com o método zscore
    """
    _, _, outliers_zscore = detect_outliers_zscore(df, variaveis, threshold)
    keep = ~outliers_zscore.any(axis=1)
    df_clean = df.loc[keep].copy()
    
    if salvar_arquivo:
        salvar(df_clean, "outliers_removed_zscore")
    
    return df_clean

def remove_outliers_iqr(df, variaveis, salvar_arquivo=True):
    """
    Remove outliers de acordo com o método IQR
    """
    _, _, outliers_por_linha_iqr = detect_outliers_iqr(df, variaveis)
    df_clean = df.loc[~outliers_por_linha_iqr].copy()
    
    if salvar_arquivo:
        salvar(df_clean, "outliers_removed_iqr")
    
    return df_clean

def remove_outliers_lof(df, variaveis, n_neighbors=20, contamination='auto', salvar_arquivo=True):
    """
    Remove outliers de acordo com o método LOF
    """
    _, outlier_labels, _ = detect_outliers_lof(df, variaveis, n_neighbors, contamination)
    df_clean = df.loc[outlier_labels == 1].copy()
    
    if salvar_arquivo:
        salvar(df_clean, "outliers_removed_lof")
    
    return df_clean

def remove_outliers(df, variaveis, method='zscore', salvar_arquivo=True, **kwargs):
    """
    Remove outliers com o método especificado.
    
    df: DataFrame.
    variaveis: Lista de colunas numéricas.
    method: 'zscore' (default), 'iqr', ou 'lof'.
    salvar_arquivo: Salva o arquivo (default: Ativado).
    **kwargs: Parâmetros adicionais para o método (threshold, n_neighbors, etc.).
    """
    if method == 'zscore':
        return remove_outliers_zscore(df, variaveis, salvar_arquivo=salvar_arquivo, **kwargs)
    elif method == 'iqr':
        return remove_outliers_iqr(df, variaveis, salvar_arquivo=salvar_arquivo)
    elif method == 'lof':
        return remove_outliers_lof(df, variaveis, salvar_arquivo=salvar_arquivo, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")