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
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from pcj.utils import salvar, separar_colunas
from pcj.modelos_ML import isolation_forest


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
# - KNN (distância ao k-ésimo vizinho);
# - STRAY;
# - Distance From The Mean.
#
# Clustering:
# - PCA --> K-Means.

# %% [markdown]
# #### Z-Score

# %%
def detect_outliers_zscore(df, variaveis=None, threshold=3, log=False, verbose=False):
    """
    Gera um quadro de outliers de um DataFrame,
    detectados pelo método Z-Score.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list, default None
        Lista de variáveis presentes em df. Se None, usa todas as colunas
        de df que não são metadados (separar_colunas).
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
    # Sem lista de variáveis: usa todas as colunas de df que não são metadados
    if variaveis is None:
        _, variaveis = separar_colunas(df)

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
def detect_outliers_iqr(df, variaveis=None, log=False, verbose=False):
    """
    Gera um quadro de outliers de um DataFrame,
    detectados pelo método IQR.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list, default None
        Lista de variáveis presentes em df. Se None, usa todas as colunas
        de df que não são metadados (separar_colunas).
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
    # Sem lista de variáveis: usa todas as colunas de df que não são metadados
    if variaveis is None:
        _, variaveis = separar_colunas(df)

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
# ##### Z-Score e IQR por mês

# %%
def detect_outliers_estatisticos(df, variaveis=None, log=True, min_variaveis=1, threshold=3,
                                 coluna_data="Data", verbose=False):
    """
    Resume por mês os métodos estatísticos (Z-Score e IQR), no mesmo formato da
    tabela do ensemble: uma linha por mês e uma coluna True/False por método.

    O Z-Score e o IQR marcam valores (mês × variável): aqui, um mês conta como
    outlier quando pelo menos `min_variaveis` variáveis foram marcadas nele.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list, default None
        Lista de variáveis presentes em df. Se None, usa todas as colunas
        de df que não são metadados (separar_colunas).
    log : bool, default True
        Repassado ao Z-Score e ao IQR: aplica log1p às variáveis antes de calcular os limites.
    min_variaveis : int, default 1
        Número mínimo de variáveis marcadas para o método marcar o mês.
    threshold : float, default 3
        Repassado ao Z-Score: |z| acima deste valor é outlier.
    coluna_data : str, default "Data"
        Nome da coluna de df com a data de cada mês.
    verbose : bool, default False
        Rodar a função exibe (print) um texto resumido dos resultados gerais.

    Returns
    -------
    tabela : pd.DataFrame
        Uma linha por mês (índice = df.index), com as colunas: a data;
        n_var_avaliadas (variáveis com valor no mês, as únicas que o Z-Score e o IQR avaliam);
        n_var_zscore e n_var_iqr (quantas variáveis cada método marcou no mês);
        Z-Score e IQR (True = o método marcou o mês).
    """
    # Sem lista de variáveis: usa todas as colunas de df que não são metadados
    if variaveis is None:
        _, variaveis = separar_colunas(df)

    # Máscaras por valor (meses × variáveis)
    _, _, mascara_zscore = detect_outliers_zscore(df, variaveis, threshold=threshold, log=log)
    _, _, mascara_iqr = detect_outliers_iqr(df, variaveis, log=log)

    # Tabela de resultados: começa com a data de cada mês
    tabela = df[[coluna_data]].copy()

    # Quantas variáveis têm valor no mês, e quantas cada método marcou (True vale 1)
    tabela["n_var_avaliadas"] = df[variaveis].notna().sum(axis=1)
    tabela["n_var_zscore"] = mascara_zscore.sum(axis=1)
    tabela["n_var_iqr"] = mascara_iqr.sum(axis=1)

    # Uma resposta por mês para cada método
    tabela["Z-Score"] = tabela["n_var_zscore"] >= min_variaveis
    tabela["IQR"] = tabela["n_var_iqr"] >= min_variaveis

    if verbose:
        escala = "log" if log else "original"
        print(f"Z-Score e IQR em escala {escala}; mês marcado com pelo menos {min_variaveis} variável(is)")
        print(f"Meses marcados: Z-Score {tabela['Z-Score'].sum()}, IQR {tabela['IQR'].sum()}")

    return tabela


# %% [markdown]
# ##### LOF

# %%
def detect_outliers_lof(df, variaveis=None, n_neighbors=20,
                        contaminacoes=(0.05, 0.06, 0.07, 0.08, 0.09, 0.10), limiar=0.5, verbose=False):
    """
    Gera um quadro de outliers de um DataFrame,
    detectados pelo método LOF (Local Outlier Factor), em grade:
    roda o LOF uma vez para cada valor de contaminação e considera
    outlier por consenso o mês marcado em pelo menos `limiar` das rodadas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list, default None
        Lista de variáveis presentes em df. Se None, usa todas as colunas
        de df que não são metadados (separar_colunas).
    n_neighbors : int, default 20
        Determina o número de vizinhos a serem considerados pelo algoritmo
        "LocalOutlierFactor" da biblioteca sklearn.
    contaminacoes : tuple of float, default (0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
        Valores de contaminação testados (uma rodada do LOF para cada um).
    limiar : float, default 0.5
        Fração mínima de rodadas para considerar o mês um outlier por consenso.
    verbose : bool, default False
        Rodar a função exibe (print) um texto resumido dos resultados gerais do algoritmo.

    Returns
    -------
    frequencia : pd.Series of float
        Fração das rodadas em que cada mês foi marcado (0 a 1). Índice = df.index (meses);
        NaN nos meses com dado faltante, que o LOF não avalia.
    consenso : pd.Series of bool
        True nos meses com frequencia >= limiar. Índice = df.index (meses);
        os meses não avaliados ficam False.
    """
    # Sem lista de variáveis: usa todas as colunas de df que não são metadados
    if variaveis is None:
        _, variaveis = separar_colunas(df)

    # O LOF não aceita NaN: só os meses completos são avaliados
    X_nonan = df[variaveis].dropna(axis=0, how='any')
    idx_nonan = X_nonan.index

    # O LOF compara distâncias entre meses: sem padronizar, as variáveis de valores grandes dominariam
    X_scaled = StandardScaler().fit_transform(X_nonan)

    # Dicionário onde as marcações de cada rodada serão armazenadas
    # Chave: contaminação. Valor: True/False para cada mês completo (True = outlier)
    marcacoes = {}

    # O LOF não tem aleatoriedade: ao contrário do Isolation Forest, não precisa de várias seeds
    for contaminacao in contaminacoes:
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contaminacao)

        # fit_predict devolve -1 (outlier) ou 1 (normal) para cada mês
        rotulos = lof.fit_predict(X_scaled)

        marcacoes[contaminacao] = rotulos == -1

    # Tabela: uma linha por mês completo, uma coluna por rodada
    marcacoes = pd.DataFrame(marcacoes, index=idx_nonan)

    # Média de True/False = fração das rodadas em que o mês foi marcado (True vale 1, False vale 0).
    # O reindex estende para todos os meses de df: os não avaliados ficam NaN
    frequencia = marcacoes.mean(axis=1).reindex(df.index)

    # NaN >= limiar dá False: os meses não avaliados não entram no consenso
    consenso = frequencia >= limiar

    if verbose:
        print(f"LOF aplicado em {len(idx_nonan)} meses completos ({len(df) - len(idx_nonan)} com NaN, não avaliados)")
        print(f"{len(contaminacoes)} contaminações; meses marcados em pelo menos {limiar:.0%} das rodadas: {consenso.sum()}")

    return frequencia, consenso


# %% [markdown]
# ##### KNN

# %%
def detect_outliers_knn(df, variaveis=None, n_neighbors=5,
                        contaminacoes=(0.05, 0.06, 0.07, 0.08, 0.09, 0.10), limiar=0.5, verbose=False):
    """
    Gera um quadro de outliers de um DataFrame,
    detectados pelo método KNN (distância ao k-ésimo vizinho mais próximo), em grade:
    para cada valor de contaminação, marca os meses mais distantes dos seus vizinhos
    e considera outlier por consenso o mês marcado em pelo menos `limiar` das rodadas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list, default None
        Lista de variáveis presentes em df. Se None, usa todas as colunas
        de df que não são metadados (separar_colunas).
    n_neighbors : int, default 5
        Número de vizinhos (k). O score de cada mês é a distância ao k-ésimo
        vizinho mais próximo (mesmo padrão do KNN da biblioteca PyOD).
    contaminacoes : tuple of float, default (0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
        Valores de contaminação testados: em cada rodada, a fração de meses
        com os maiores scores é marcada como outlier.
    limiar : float, default 0.5
        Fração mínima de rodadas para considerar o mês um outlier por consenso.
    verbose : bool, default False
        Rodar a função exibe (print) um texto resumido dos resultados gerais do algoritmo.

    Returns
    -------
    frequencia : pd.Series of float
        Fração das rodadas em que cada mês foi marcado (0 a 1). Índice = df.index (meses);
        NaN nos meses com dado faltante, que o KNN não avalia.
    consenso : pd.Series of bool
        True nos meses com frequencia >= limiar. Índice = df.index (meses);
        os meses não avaliados ficam False.
    """
    # Sem lista de variáveis: usa todas as colunas de df que não são metadados
    if variaveis is None:
        _, variaveis = separar_colunas(df)

    # O KNN não aceita NaN: só os meses completos são avaliados
    X_nonan = df[variaveis].dropna(axis=0, how='any')
    idx_nonan = X_nonan.index

    # O KNN mede distâncias entre meses: sem padronizar, as variáveis de valores grandes dominariam
    X_scaled = StandardScaler().fit_transform(X_nonan)

    # Pede n_neighbors + 1 vizinhos porque o vizinho mais próximo de cada mês é ele mesmo (distância 0)
    vizinhos = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_scaled)
    distancias_vizinhos, _ = vizinhos.kneighbors(X_scaled)

    # Score: distância de cada mês ao seu k-ésimo vizinho (a coluna 0 é o próprio mês)
    score = pd.Series(distancias_vizinhos[:, n_neighbors], index=idx_nonan)

    # Dicionário onde as marcações de cada rodada serão armazenadas
    # Chave: contaminação. Valor: True/False para cada mês completo (True = outlier)
    marcacoes = {}

    # O KNN não tem aleatoriedade: o score é calculado uma vez, e só o corte muda a cada contaminação
    for contaminacao in contaminacoes:
        # Corte: são marcados os meses com score acima do percentil (1 - contaminação)
        corte = np.percentile(score, 100 * (1 - contaminacao))
        marcacoes[contaminacao] = score > corte

    # Tabela: uma linha por mês completo, uma coluna por rodada
    marcacoes = pd.DataFrame(marcacoes, index=idx_nonan)

    # Média de True/False = fração das rodadas em que o mês foi marcado (True vale 1, False vale 0).
    # O reindex estende para todos os meses de df: os não avaliados ficam NaN
    frequencia = marcacoes.mean(axis=1).reindex(df.index)

    # NaN >= limiar dá False: os meses não avaliados não entram no consenso
    consenso = frequencia >= limiar

    if verbose:
        print(f"KNN aplicado em {len(idx_nonan)} meses completos ({len(df) - len(idx_nonan)} com NaN, não avaliados)")
        print(f"{len(contaminacoes)} contaminações; meses marcados em pelo menos {limiar:.0%} das rodadas: {consenso.sum()}")

    return frequencia, consenso


# %% [markdown]
# ##### PCA --> K-Means

# %%
def detect_outliers_pca_kmeans(df, variaveis=None, n_clusters=3,
                               contaminacoes=(0.05, 0.06, 0.07, 0.08, 0.09, 0.10), n_seeds=30,
                               limiar=0.5, verbose=False):
    """
    Gera um quadro de outliers de um DataFrame,
    usando PCA + K-Means em grade: para cada seed, agrupa os meses (K-Means sobre os
    componentes principais) e mede a distância de cada mês ao centro do seu grupo;
    para cada valor de contaminação, marca os meses mais distantes. É outlier por
    consenso o mês marcado em pelo menos `limiar` das rodadas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list, default None
        Lista de variáveis presentes em df. Se None, usa todas as colunas
        de df que não são metadados (separar_colunas).
    n_clusters : int, default 3
        Determina o número de clusters a serem utilizados pelo algoritmo
        "KMeans" da biblioteca sklearn.
    contaminacoes : tuple of float, default (0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
        Valores de contaminação testados: em cada rodada, a fração de meses
        mais distantes do centro do seu grupo é marcada como outlier.
    n_seeds : int, default 30
        Número de seeds do K-Means (random_state de 0 a n_seeds - 1): a posição
        inicial dos centros é aleatória e pode mudar os grupos.
    limiar : float, default 0.5
        Fração mínima de rodadas para considerar o mês um outlier por consenso.
    verbose : bool, default False
        Rodar a função exibe (print) um texto resumido dos resultados gerais do algoritmo.

    Returns
    -------
    frequencia : pd.Series of float
        Fração das rodadas em que cada mês foi marcado (0 a 1). Índice = df.index (meses);
        NaN nos meses com dado faltante, que o K-Means não avalia.
    consenso : pd.Series of bool
        True nos meses com frequencia >= limiar. Índice = df.index (meses);
        os meses não avaliados ficam False.
    """
    # Sem lista de variáveis: usa todas as colunas de df que não são metadados
    if variaveis is None:
        _, variaveis = separar_colunas(df)

    # O PCA e o K-Means não aceitam NaN: só os meses completos são avaliados
    X_nonan = df[variaveis].dropna(axis=0, how='any')
    idx_nonan = X_nonan.index

    # Padronização: sem ela, as variáveis de valores grandes dominariam as distâncias
    X_scaled = StandardScaler().fit_transform(X_nonan)

    # Componentes principais que explicam 95% da variância.
    # O PCA não depende de seed aqui: é calculado uma vez, fora do laço
    X_pca = PCA(n_components=0.95).fit_transform(X_scaled)

    # Dicionário onde as marcações de cada rodada serão armazenadas
    # Chave: (contaminação, seed). Valor: True/False para cada mês completo (True = outlier)
    marcacoes = {}

    for seed in range(n_seeds):
        # Os grupos dependem só da seed: o K-Means roda uma vez por seed, e só o corte muda a cada contaminação
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        grupos = kmeans.fit_predict(X_pca)

        # Distância de cada mês ao centro do seu grupo
        distancias = np.linalg.norm(X_pca - kmeans.cluster_centers_[grupos], axis=1)

        for contaminacao in contaminacoes:
            # Corte: são marcados os meses com distância acima do percentil (1 - contaminação)
            corte = np.percentile(distancias, 100 * (1 - contaminacao))
            marcacoes[(contaminacao, seed)] = distancias > corte

    # Tabela: uma linha por mês completo, uma coluna por rodada
    marcacoes = pd.DataFrame(marcacoes, index=idx_nonan)

    # Média de True/False = fração das rodadas em que o mês foi marcado (True vale 1, False vale 0).
    # O reindex estende para todos os meses de df: os não avaliados ficam NaN
    frequencia = marcacoes.mean(axis=1).reindex(df.index)

    # NaN >= limiar dá False: os meses não avaliados não entram no consenso
    consenso = frequencia >= limiar

    if verbose:
        print(f"PCA + K-Means aplicado em {len(idx_nonan)} meses completos ({len(df) - len(idx_nonan)} com NaN, "
              f"não avaliados); {X_pca.shape[1]} componentes principais")
        print(f"{len(contaminacoes)} contaminações x {n_seeds} seeds = {marcacoes.shape[1]} rodadas; "
              f"meses marcados em pelo menos {limiar:.0%} das rodadas: {consenso.sum()}")

    return frequencia, consenso



# %% [markdown]
# ##### Ensemble (IF + LOF + KNN; K-Means opcional)

# %%
def detect_outliers_ensemble(df, variaveis=None, metodos=("IF", "LOF", "KNN"), regra="maioria",
                             contaminacoes=(0.05, 0.06, 0.07, 0.08, 0.09, 0.10), limiar=0.5, n_seeds=30,
                             n_clusters=3, limiar_media=0.5, coluna_data="Data", verbose=False):
    """
    Combina detectores de outliers de aprendizado de máquina (Isolation Forest,
    LOF, KNN e, como opção, PCA + K-Means) num ensemble. Cada detector vota com o
    seu consenso (ver pcj.modelos_ML.isolation_forest, detect_outliers_lof,
    detect_outliers_knn e detect_outliers_pca_kmeans), e a regra de votação é
    aplicada só entre os detectores que avaliaram o mês (o LOF, o KNN e o K-Means
    não avaliam os meses com NaN).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list, default None
        Lista de variáveis presentes em df. Se None, usa todas as colunas
        de df que não são metadados (separar_colunas).
    metodos : tuple of str, default ("IF", "LOF", "KNN")
        Detectores que votam no ensemble: "IF", "LOF", "KNN" e, fora do padrão,
        "K-Means" (PCA + K-Means).
    regra : str, default "maioria"
        Como os votos viram a marcação do ensemble:
        "maioria": mais da metade dos detectores que avaliaram o mês
        (com número par de detectores, o empate não marca);
        "unanimidade": todos os detectores que avaliaram o mês;
        "qualquer": pelo menos um detector;
        "media": média das frequências dos detectores (voto suave) >= limiar_media.
    contaminacoes : tuple of float, default (0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
        Valores de contaminação repassados aos detectores.
    limiar : float, default 0.5
        Limiar de consenso de cada detector (fração mínima de rodadas).
    n_seeds : int, default 30
        Número de seeds do Isolation Forest e do K-Means.
    n_clusters : int, default 3
        Número de grupos do K-Means (só usado quando "K-Means" está em metodos).
        Ver pcj.modelos_ML.selecao_k para a escolha.
    limiar_media : float, default 0.5
        Só para regra="media": média mínima das frequências para marcar o mês.
    coluna_data : str, default "Data"
        Nome da coluna de df com a data de cada mês.
    verbose : bool, default False
        Rodar a função exibe (print) um texto resumido dos resultados gerais do algoritmo.

    Returns
    -------
    tabela : pd.DataFrame
        Uma linha por mês (índice = df.index), com as colunas: a data; uma coluna por
        detector (True = o detector marcou o mês); avaliaram (quantos detectores
        avaliaram o mês); votos (quantos o marcaram); media (média das frequências
        entre os que avaliaram); ensemble (marcação final, segundo a regra).

    Raises
    ------
    ValueError
        Se metodos tiver um detector desconhecido ou se regra não for uma das opções.
    """
    # Sem lista de variáveis: usa todas as colunas de df que não são metadados
    if variaveis is None:
        _, variaveis = separar_colunas(df)

    # Tabela de resultados: começa com a data de cada mês
    tabela = df[[coluna_data]].copy()

    # Frequência de cada detector, uma coluna por detector (NaN = o detector não avaliou o mês)
    frequencias = pd.DataFrame(index=df.index)

    for metodo in metodos:
        if metodo == "IF":
            frequencia, consenso = isolation_forest(df, variaveis, contaminacoes=contaminacoes,
                                                    n_seeds=n_seeds, limiar=limiar)
        elif metodo == "LOF":
            frequencia, consenso = detect_outliers_lof(df, variaveis, contaminacoes=contaminacoes, limiar=limiar)
        elif metodo == "KNN":
            frequencia, consenso = detect_outliers_knn(df, variaveis, contaminacoes=contaminacoes, limiar=limiar)
        elif metodo == "K-Means":
            frequencia, consenso = detect_outliers_pca_kmeans(df, variaveis, n_clusters=n_clusters,
                                                              contaminacoes=contaminacoes,
                                                              n_seeds=n_seeds, limiar=limiar)
        else:
            raise ValueError(f"Método desconhecido: {metodo}. Use 'IF', 'LOF', 'KNN' ou 'K-Means'.")

        tabela[metodo] = consenso
        frequencias[metodo] = frequencia

    # Quantos detectores avaliaram cada mês e quantos o marcaram (True vale 1).
    # list(metodos): com uma tupla entre colchetes, o pandas procuraria uma coluna só com esse nome
    tabela["avaliaram"] = frequencias.notna().sum(axis=1)
    tabela["votos"] = tabela[list(metodos)].sum(axis=1)

    # Média das frequências entre os detectores que avaliaram o mês (o mean ignora os NaN)
    tabela["media"] = frequencias.mean(axis=1)

    if regra == "maioria":
        # "Mais da metade" sem divisão: evita dividir por 0 quando nenhum detector avaliou o mês
        tabela["ensemble"] = tabela["votos"] * 2 > tabela["avaliaram"]
    elif regra == "unanimidade":
        tabela["ensemble"] = (tabela["votos"] == tabela["avaliaram"]) & (tabela["avaliaram"] > 0)
    elif regra == "qualquer":
        tabela["ensemble"] = tabela["votos"] >= 1
    elif regra == "media":
        # NaN >= limiar dá False: mês que nenhum detector avaliou não é marcado
        tabela["ensemble"] = tabela["media"] >= limiar_media
    else:
        raise ValueError(f"Regra desconhecida: {regra}. Use 'maioria', 'unanimidade', 'qualquer' ou 'media'.")

    if verbose:
        print(f"Ensemble ({' + '.join(metodos)}, regra '{regra}'): {tabela['ensemble'].sum()} meses marcados")
        print(f"Meses avaliados por um único detector: {(tabela['avaliaram'] == 1).sum()}")

    return tabela


# %% [markdown]
# ##### Comparação de marcações

# %%
def comparar_marcacoes(tabela, referencia="ensemble", metodos=None):
    """
    Compara marcações por mês (colunas True/False de uma tabela) com uma
    marcação de referência: quantos meses cada método marcou, quantos em comum
    com a referência e o índice de Jaccard.

    Serve para qualquer combinação: métodos × ensemble, detectores × ensemble
    ou um cenário do ensemble × outro.

    Parameters
    ----------
    tabela : pd.DataFrame
        Uma linha por mês, com colunas True/False (ex.: a tabela de
        detect_outliers_ensemble, a de detect_outliers_estatisticos ou uma
        montada com colunas de várias delas).
    referencia : str, default "ensemble"
        Coluna de tabela usada como referência.
    metodos : list of str, default None
        Colunas de tabela comparadas com a referência. Se None, usa todas as
        colunas True/False de tabela, menos a referência.

    Returns
    -------
    resumo : pd.DataFrame
        Uma linha por método, com as colunas: marcados (meses que o método marcou),
        em_comum (marcados pelo método e pela referência), so_no_metodo,
        so_na_referencia e jaccard (em_comum ÷ meses marcados por pelo menos um
        dos dois: 1 = marcaram os mesmos meses; 0 = nenhum em comum).

    Raises
    ------
    ValueError
        Se a referência ou algum dos métodos não for uma coluna True/False de tabela.
    """
    # Sem lista de métodos: todas as colunas True/False, menos a referência
    if metodos is None:
        metodos = [coluna for coluna in tabela.columns if tabela[coluna].dtype == bool and coluna != referencia]

    # Uma coluna que não é True/False (ex.: com NaN depois de juntar tabelas de índices diferentes)
    # daria contas erradas sem avisar: com ela, o ~ não funciona como "não"
    for coluna in [referencia] + list(metodos):
        if coluna not in tabela.columns or tabela[coluna].dtype != bool:
            raise ValueError(f"{coluna!r} precisa ser uma coluna True/False de tabela.")

    marcacao_referencia = tabela[referencia]

    # Dicionário para armazenar a comparação de cada método com a referência
    resumo = {}

    for metodo in metodos:
        marcacao = tabela[metodo]

        # & = "e", | = "ou", ~ = "não", aplicados mês a mês
        em_comum = (marcacao & marcacao_referencia).sum()
        algum_dos_dois = (marcacao | marcacao_referencia).sum()

        resumo[metodo] = {
            "marcados": marcacao.sum(),
            "em_comum": em_comum,
            "so_no_metodo": (marcacao & ~marcacao_referencia).sum(),
            "so_na_referencia": (~marcacao & marcacao_referencia).sum(),
            # Sem nenhum mês marcado pelos dois, a divisão seria 0 ÷ 0
            "jaccard": em_comum / algum_dos_dois if algum_dos_dois > 0 else np.nan,
        }

    # Conversão do dicionário de resultados num DataFrame
    return pd.DataFrame.from_dict(resumo, orient="index")


# %% [markdown]
# ##### Todos

# %%
def detect_outliers(df, variaveis=None, methods=None):
    """
    Executa múltiplos métodos de detecção de outliers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list, default None
        Lista de variáveis presentes em df. Se None, usa todas as colunas
        de df que não são metadados (separar_colunas).
    methods : list of str, default None
        Métodos de detecção a executar. Se ``None``, todos os métodos
        disponíveis (Z-Score, IQR, LOF, KNN, PCA + K-Means) são executados.

    Returns
    -------
    results : dict
        Armazena os resultados do(s) método(s) de detecção utilizados.
    """    
    if methods is None:
        methods = ['zscore', 'iqr', 'lof', 'knn', 'pca_kmeans']
    
    results = {}
    
    if 'zscore' in methods:
        results['zscore'] = detect_outliers_zscore(df, variaveis)
    
    if 'iqr' in methods:
        results['iqr'] = detect_outliers_iqr(df, variaveis)
    
    if 'lof' in methods:
        results['lof'] = detect_outliers_lof(df, variaveis)

    if 'knn' in methods:
        results['knn'] = detect_outliers_knn(df, variaveis)
    
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
def remove_outliers_zscore(df, variaveis=None, threshold=3, salvar_arquivo=True):
    """
    Remove outliers de acordo com o método Z-Score.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list, default None
        Lista de variáveis presentes em df. Se None, usa todas as colunas
        de df que não são metadados (separar_colunas).
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
def remove_outliers_iqr(df, variaveis=None, salvar_arquivo=True):
    """
    Remove outliers de acordo com o método IQR (Interquartile Range).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list, default None
        Lista de variáveis presentes em df. Se None, usa todas as colunas
        de df que não são metadados (separar_colunas).
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
def remove_outliers_lof(df, variaveis=None, n_neighbors=20,
                        contaminacoes=(0.05, 0.06, 0.07, 0.08, 0.09, 0.10), limiar=0.5, salvar_arquivo=True):
    """
    Remove outliers de acordo com o método LOF (Local Outlier Factor),
    usando o consenso da grade de contaminações (ver detect_outliers_lof).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list, default None
        Lista de variáveis presentes em df. Se None, usa todas as colunas
        de df que não são metadados (separar_colunas).
    n_neighbors : int, default 20
        Determina o número de vizinhos a serem considerados pelo algoritmo
        "LocalOutlierFactor" da biblioteca sklearn.
    contaminacoes : tuple of float, default (0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
        Valores de contaminação testados (uma rodada do LOF para cada um).
    limiar : float, default 0.5
        Fração mínima de rodadas para considerar o mês um outlier por consenso.
    salvar_arquivo : bool, default True
        Determina se um arquivo Excel do DataFrame com os outliers removidos deve ser salvo.

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame sem os meses marcados por consenso. Os meses com NaN,
        que o LOF não avalia, são mantidos.
    """
    _, consenso = detect_outliers_lof(df, variaveis, n_neighbors, contaminacoes, limiar)
    df_clean = df.loc[~consenso].copy()

    if salvar_arquivo:
        salvar(df_clean, "outliers_removed_lof")

    return df_clean


# %% [markdown]
# ##### Todos

# %%
def remove_outliers(df, variaveis=None, method='zscore', salvar_arquivo=True, **kwargs):
    """
    Remove outliers com o método especificado.
    Usa as funções previamente definidas para
    realizar a remoção.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    variaveis : list, default None
        Lista de variáveis presentes em df. Se None, usa todas as colunas
        de df que não são metadados (separar_colunas).
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


# %% [markdown]
# # Tratamento de Outliers
#
# Em vez de remover o mês inteiro, troca por NaN só os valores marcados, nos meses marcados.
# O mês continua na base, e a imputação preenche os valores apagados depois.

# %%
def tratar_outliers(df, marcacao_meses, mascara_valores, coluna_data="Data", verbose=False):
    """
    Trata outliers valor a valor: troca por NaN os valores marcados por
    mascara_valores, mas só nos meses marcados por marcacao_meses. O mês
    continua na base; os valores apagados ficam para a imputação.

    Uso típico: marcacao_meses vem de um método multivariado, que diz QUAL MÊS
    é estranho (ex.: a coluna "ensemble" de detect_outliers_ensemble), e
    mascara_valores vem de um univariado, que diz QUAL VARIÁVEL (ex.: a máscara
    de detect_outliers_iqr). Para tratar todos os valores da máscara, sem filtro
    de meses, use marcacao_meses = pd.Series(True, index=df.index).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser tratado.
    marcacao_meses : pd.Series of bool
        Uma resposta por mês (índice = df.index): True = mês marcado.
    mascara_valores : pd.DataFrame of bool
        Máscara por valor (índice = df.index; colunas = variáveis): True = valor marcado.
        As colunas da máscara definem as variáveis tratadas.
    coluna_data : str, default "Data"
        Nome da coluna de df com a data de cada mês.
    verbose : bool, default False
        Rodar a função exibe (print) um texto resumido dos resultados.

    Returns
    -------
    df_tratado : pd.DataFrame
        Cópia de df com os valores tratados trocados por NaN.
    registro : pd.DataFrame
        Uma linha por valor trocado por NaN, com a data do mês, a variável
        e o valor original. Documenta o tratamento (ex.: para investigar se
        os valores são erros ou eventos reais).

    Raises
    ------
    ValueError
        Se marcacao_meses ou mascara_valores tiverem índice diferente do de df,
        ou não forem True/False.
    """
    # Com índices diferentes, o pandas alinharia as linhas erradas (ou criaria NaN) sem avisar
    if not (marcacao_meses.index.equals(df.index) and mascara_valores.index.equals(df.index)):
        raise ValueError("marcacao_meses e mascara_valores precisam ter o mesmo índice de df.")

    # Com NaN ou números no lugar de True/False, o ~ e o .loc abaixo dariam resultados errados
    if marcacao_meses.dtype != bool or not (mascara_valores.dtypes == bool).all():
        raise ValueError("marcacao_meses e mascara_valores precisam ser True/False.")

    # As colunas da máscara definem as variáveis tratadas
    variaveis = list(mascara_valores.columns)

    # Valores a tratar: marcados pela máscara E num mês marcado.
    # Começa da máscara e desliga os meses não marcados: neles, nada é tratado.
    # (mascara_valores & marcacao_meses não serviria: o pandas alinharia a Series com as colunas, não com as linhas)
    tratar = mascara_valores.copy()
    tratar.loc[~marcacao_meses] = False

    # Cópia de df com os valores tratados trocados por NaN (o mask troca onde tratar é True)
    df_tratado = df.copy()
    df_tratado[variaveis] = df[variaveis].mask(tratar)

    # Registro: uma linha por valor trocado por NaN, com o valor original
    linhas = []
    for indice, linha in tratar.iterrows():
        # linha.index[linha]: nomes das variáveis que estão True nesse mês (como em resumo_por_mes)
        for variavel in linha.index[linha]:
            linhas.append({
                coluna_data: df.at[indice, coluna_data],
                "variavel": variavel,
                "valor_original": df.at[indice, variavel],
            })
    registro = pd.DataFrame(linhas, columns=[coluna_data, "variavel", "valor_original"])

    if verbose:
        meses_tratados = tratar.any(axis=1)
        sem_valor = marcacao_meses & ~meses_tratados
        print(f"Valores trocados por NaN: {len(registro)}, em {meses_tratados.sum()} meses")
        print(f"Meses marcados sem nenhum valor marcado (ficam intactos): {sem_valor.sum()}")

    return df_tratado, registro
