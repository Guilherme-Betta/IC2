# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from pcj.utils import salvar


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
def detect_outliers_zscore(df, variaveis, threshold=3, log=False, verbose=False):
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
    log : bool, default False
        Transforma o formato de distribuição dos dados de forma logarítmica,
        tornando-a mais simétrica.
    verbose : bool, default False
        Rodar a função exibe (print) um texto resumido dos resultados gerais do algoritmo.

    Returns
    -------
    limites_z : pd.DataFrame
        Uma linha por variável: limites inferior e superior (média ± threshold × desvio
        padrão, na unidade original) e número de outliers.
    total_outliers_z : int
        Número de linhas com pelo menos um outlier.
    outliers_zscore : pd.DataFrame of bool
        Máscara booleana indicando se cada valor foi classificado como outlier.
        Índice = df.index (meses); colunas = variaveis.
    """    
    # Cópia dos dados para preservar o dataframe original
    df = df.copy()

    # Transforma o formato de distribuição dos dados de forma logarítmica, tornando-a mais simétrica.
    if log:
        df[variaveis] = np.log1p(df[variaveis])

    # Calcular Z-Scores para todas as colunas numéricas
    z_scores = zscore(df[variaveis], nan_policy='omit')     # nan_policy='omit': ignora os NaN no cálculo da média e do desvio padrão.
                                                            # Sem isso, uma coluna com um único NaN tem todos os z-scores = NaN e 
                                                            # nenhum outlier é marcado.

    # Identificar outliers: |z| > threshold (3 é um threshold comum)
    # DataFrame com os mesmos meses (índice) e variáveis (colunas) de df: mostra ONDE estão os outliers
    outliers_zscore = pd.DataFrame(np.abs(z_scores) > threshold, index=df.index, columns=variaveis)

    # Dicionário para armazenar os limites e a contagem de outliers de cada variável
    limites = {}

    for coluna in variaveis:
        # Mesma média e desvio padrão que o zscore usa (ddof=0; o .std() do pandas usa ddof=1 por padrão)
        media = df[coluna].mean()
        desvio = df[coluna].std(ddof=0)

        # Definição dos limites dos dados normais
        limite_inferior = media - threshold * desvio
        limite_superior = media + threshold * desvio

        # Converte os limites de volta para a escala original ANTES de guardar no dicionário
        if log:
            limite_inferior, limite_superior = np.expm1(limite_inferior), np.expm1(limite_superior)

        limites[coluna] = {
            "limite_inferior": limite_inferior,
            "limite_superior": limite_superior,
            "n_outliers": outliers_zscore[coluna].sum(),   # contagem vem da grade
        }

    # Conversão do dicionário de resultados num DataFrame
    limites_z = pd.DataFrame.from_dict(limites, orient="index")

    # Total de linhas com pelo menos um outlier (calculado antes do verbose, que usa esse valor)
    total_outliers_z = outliers_zscore.any(axis=1).sum()

    if verbose:
        print(f"Total de outliers (Z-Score, somando todas as colunas): {limites_z['n_outliers'].sum()}")
        print(f"Total de linhas com pelo menos um outlier (Z-Score): {total_outliers_z}")

    return limites_z, total_outliers_z, outliers_zscore



# %% [markdown]
# ##### IQR

# %%
def detect_outliers_iqr(df, variaveis, log=False, verbose=False):
    """
    Gera um quadro de outliers de um DataFrame,
    detectados pelo método IQR.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list
        Lista de variáveis presentes em df.
    log : bool, default False
        Transforma o formato de distribuição dos dados de forma logarítmica,
        tornando-a mais simétrica.
    verbose : bool, default False
        Rodar a função exibe (print) um texto resumido dos resultados gerais do algoritmo.

    Returns
    -------
    limites_iqr : pd.DataFrame
        DataFrame contendo os resultados do algoritmo para cada variável.
    total_linhas_outliers_iqr : int
        Número total de linhas com ao menos um outlier.
    outliers_iqr : pd.DataFrame of bool
        Máscara booleana indicando se cada valor foi classificado como outlier.
        Índice = df.index (meses); colunas = variaveis.
    """       
    # Cópia dos dados para preservar o dataframe original
    df = df.copy()

    # Transforma o formato de distribuição dos dados de forma logarítmica, tornando-a mais simétrica.
    if log:
        df[variaveis] = np.log1p(df[variaveis])
    
    # Dicionário para armazenar os resultados do algoritmo
    limites = {}

    # Máscara: DataFrame com os mesmos meses (índice) e variáveis (colunas) de df: mostra ONDE estão os outliers
    outliers_iqr = pd.DataFrame(False, index=df.index, columns=variaveis)
    
    for coluna in variaveis:
        # Definição dos Quartis
        Q1 = df[coluna].quantile(0.25)
        Q3 = df[coluna].quantile(0.75)

        # Definição do Intervalo Interquartil
        IQR = Q3 - Q1

        # Definição dos limites dos dados normais
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        # Identificação de Outliers
        outliers = (df[coluna] < limite_inferior) | (df[coluna] > limite_superior)

        # Converte os limites de volta para a escala original, facilitando a leitura
        if log:
            limite_inferior, limite_superior = np.expm1(limite_inferior), np.expm1(limite_superior)

        limites[coluna] = {
        "limite_inferior": limite_inferior,
        "limite_superior": limite_superior,
        "n_outliers": outliers.sum(),
        }

        # Armazenamento da Posição dos Outliers
        outliers_iqr[coluna] = outliers

    # Cálculo do total de linhas com outliers
    total_linhas_outliers_iqr = outliers_iqr.any(axis=1).sum()

    # Conversão do dicionário de resultados num DataFrame
    limites_iqr = pd.DataFrame.from_dict(limites, orient="index")

    if verbose:
        print(f"\nTotal de outliers (IQR, somando todas as colunas): {limites_iqr['n_outliers'].sum()}")
        print(f"Total de linhas com pelo menos um outlier (IQR): {total_linhas_outliers_iqr}")
    
    return limites_iqr, total_linhas_outliers_iqr, outliers_iqr


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
def detect_outliers_pca_kmeans(df, variaveis, n_clusters=3, random_state=42, verbose=False):
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
    verbose : bool, default False
        Rodar a função exibe (print) um texto resumido dos resultados gerais do algoritmo.

    Returns
    -------
    num_outliers : int
        Armazena o número de outliers identificados em df.
    limiar_outlier : float
        Armazena o valor de distância máximo para que
        um dado seja considerado um outlier.
    distancias : pd.Series of float
        Distância de cada mês ao centro do seu cluster. Índice = df.index (meses);
        NaN nos meses com dado faltante, que o K-Means não avalia.
    outliers_kmeans : pd.Series of bool
        True nos meses com distância acima do limiar. Índice = df.index (meses);
        os meses não avaliados ficam False.
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

    # Distância de cada mês ao centro do seu cluster, rotulada com o índice dos meses completos
    distancias = pd.Series(index=idx_nonan, data=np.linalg.norm(X_pca - kmeans.cluster_centers_[cluster_labels], axis=1))
    
    # Limiar calculado ANTES do reindex: com NaN, o np.percentile devolve NaN e nenhum mês seria marcado
    limiar_outlier = np.percentile(distancias, 95)

    # Estende para todos os meses de df. O reindex devolve uma Series nova (por isso o "distancias =").
    # Os meses com NaN ficam com distância NaN (não avaliados)
    distancias = distancias.reindex(df.index)

    # Máscara por mês (uma resposta por mês, não por variável). NaN > limiar dá False,
    # então os meses não avaliados ficam False
    outliers_kmeans = distancias > limiar_outlier
    num_outliers = outliers_kmeans.sum()

    if verbose:
        print('K-Means aplicado em', len(idx_nonan), 'pontos. Clusters:', np.unique(cluster_labels))
        print('Número de outliers K-Means (dist > 95%):', num_outliers)
        print('Limiar de distância:', limiar_outlier)
    
    return num_outliers, limiar_outlier, distancias, outliers_kmeans



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


# %%
def resumo_por_mes(df, mascara, coluna_data="Data"):
    """
    Resume uma máscara de outliers (meses × variáveis) por mês:
    lista os meses com pelo menos uma variável marcada, quantas e quais.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame analisado (o mesmo usado para gerar a máscara).
    mascara : pd.DataFrame of bool
        Máscara devolvida por detect_outliers_zscore ou detect_outliers_iqr
        (índice = df.index; colunas = variaveis).
    coluna_data : str, default "Data"
        Nome da coluna de df com a data de cada mês.

    Returns
    -------
    tabela : pd.DataFrame
        Uma linha por mês marcado, com a data, o número de variáveis marcadas
        (n_variaveis) e os nomes delas (variaveis), do mês com mais variáveis
        para o com menos.
    """
    # Um mês entra se pelo menos uma variável estiver marcada
    meses = mascara.any(axis=1)

    tabela = df.loc[meses, [coluna_data]].assign(
        # Quantas variáveis foram marcadas naquele mês (True vale 1)
        n_variaveis=mascara.sum(axis=1)[meses],
        # Quais variáveis: para cada linha, junta os nomes das colunas que estão True
        variaveis=mascara[meses].apply(lambda linha: ", ".join(linha.index[linha]), axis=1),
    )

    return tabela.sort_values("n_variaveis", ascending=False)



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
    _, _, outliers_iqr = detect_outliers_iqr(df, variaveis)
    df_clean = df.loc[~outliers_iqr.any(axis=1)].copy()
    
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
