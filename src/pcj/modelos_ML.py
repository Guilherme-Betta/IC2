# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # Isolation Forest - Detecção de Anomalias

# %%
def isolation_forest(
    df,
    variaveis=None,
    contaminacoes=(0.05, 0.06, 0.07, 0.08, 0.09, 0.10),
    n_seeds=30,
    limiar=0.5,
    verbose=False,
):
    """
    Roda o Isolation Forest em grade: cada valor de contaminação × várias seeds.
    Conta em que fração das rodadas cada linha (mês) foi marcada como anomalia
    e considera anomalia por consenso a linha marcada em pelo menos `limiar` das rodadas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse
    variaveis : list, optional
        Lista com nomes das colunas identificadas como variáveis de df, by default None
    contaminacoes : tuple of float, optional
        Valores de contaminação testados, by default (0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
    n_seeds : int, optional
        Número de seeds (random_state de 0 a n_seeds - 1) por contaminação, by default 30
    limiar : float, optional
        Fração mínima de rodadas para considerar a linha uma anomalia por consenso, by default 0.5
    verbose : bool, optional
        Se True, exibe (print) um resumo dos resultados, by default False

    Returns
    -------
    frequencia : pd.Series
        Fração das rodadas em que cada linha foi marcada (0 a 1), com o índice de df.
    consenso : pd.Series of bool
        True para as linhas com frequencia >= limiar.
    """
    # Bibliotecas
    import pandas as pd
    from sklearn.ensemble import IsolationForest

    # Separação de variaveis
    if variaveis is None:
        from pcj.utils import separar_colunas
        _, variaveis = separar_colunas(df)

    # Base de dados a ser utilizada
    X = df[variaveis].copy()

    # Dicionário onde as marcações de cada rodada serão armazenadas
    # Chave: (contaminação, seed). Valor: True/False para cada linha (True = anomalia)
    marcacoes = {}

    for contaminacao in contaminacoes:
        for seed in range(n_seeds):
            modelo = IsolationForest(contamination=contaminacao, random_state=seed)

            # fit_predict treina o modelo e devolve -1 (anomalia) ou 1 (normal) para cada linha.
            # O corte já é feito pela contaminação: substitui o cálculo manual com np.percentile
            rotulos = modelo.fit_predict(X)

            marcacoes[(contaminacao, seed)] = rotulos == -1

    # Tabela: uma linha por mês, uma coluna por rodada
    marcacoes = pd.DataFrame(marcacoes, index=df.index)

    # Média de True/False = fração das rodadas em que a linha foi marcada (True vale 1, False vale 0)
    frequencia = marcacoes.mean(axis=1)

    consenso = frequencia >= limiar

    if verbose:
        print(f"{len(contaminacoes)} contaminações x {n_seeds} seeds = {marcacoes.shape[1]} rodadas")
        print(f"Linhas marcadas em pelo menos {limiar:.0%} das rodadas: {consenso.sum()}")

    return frequencia, consenso



# %% [markdown]
# # Random Forest - Regressão

# %%
def random_forest(df, variaveis=None, var_alvo='pH', n_blocos=5, embaralhar=False):
    """
    Roda um algoritmo de regressão com o modelo Random Forest.
    Avalia o modelo com validação cruzada: divide os meses em n_blocos, e cada bloco é testado
    uma vez, com o modelo treinado nos outros blocos.
    Testa a qualidade das previsões do modelo com as métricas de avaliação "MAE", "RMSE" e "R²",
    juntando as previsões de todos os blocos, e com o R² de cada bloco.
    Imprime esses resultados no final.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse
    variaveis : list, optional
        Lista com nomes das colunas identificadas como variaveis de df, by default None
    var_alvo : str, optional
        Variável alvo da regressão, by default 'pH'
    n_blocos : int, optional
        Número de blocos da validação cruzada, by default 5
    embaralhar : bool, optional
        Se False, cada bloco é um período de meses seguidos (a base está em ordem de data),
        e o modelo é testado em anos que não viu. Se True, os meses são sorteados para os
        blocos (KFold aleatório, random_state=42), by default False
    """    
    X = df.copy()

    if variaveis is None:
                from pcj.utils import separar_colunas
                _, variaveis = separar_colunas(df)

    # Carregar e preparar dados

    X_nonan = X.dropna().copy() # Random Forest não funciona quando há NaN presentes.

    var_alvo = var_alvo

    x = X_nonan[variaveis].drop(var_alvo, axis=1) 
    y = X_nonan[var_alvo]

    # Divisão dos meses em blocos para a validação cruzada. Sem embaralhar, o KFold corta a base
    # na ordem das linhas (ordem de data): cada bloco é um período de meses seguidos
    from sklearn.model_selection import KFold

    if embaralhar:
        kfold = KFold(n_splits=n_blocos, shuffle=True, random_state=42)
    else:
        kfold = KFold(n_splits=n_blocos)

    # Bibliotecas do modelo e das métricas
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

    # Previsão de cada mês, feita quando o bloco dele é o de teste
    y_pred = pd.Series(index=y.index, dtype=float)
    r2_blocos = []

    # Cada volta do laço usa um bloco como teste e os outros como treino
    for treino, teste in kfold.split(x):
        x_train, x_test = x.iloc[treino], x.iloc[teste]
        y_train, y_test = y.iloc[treino], y.iloc[teste]

        # Criando o modelo
        model = RandomForestRegressor(n_estimators=100,    # Copiando do tutorial
                                    random_state=42)

        # Treinando o modelo
        model.fit(x_train, y_train)

        # Previsões do bloco de teste
        y_pred.iloc[teste] = model.predict(x_test)

        r2_blocos.append(r2_score(y_test, y_pred.iloc[teste]))

    # Avaliação do Desempenho, juntando as previsões de todos os blocos
    print("RMSE:", root_mean_squared_error(y, y_pred))
    print("MAE:", mean_absolute_error(y, y_pred))
    print("r²:", r2_score(y, y_pred))

    # Validação Cruzada: R² de cada bloco e média ± DP dos blocos
    r2_blocos = pd.Series(r2_blocos)
    print("R² de cada bloco:", r2_blocos.round(3).tolist())
    print(f"R² dos blocos: média {r2_blocos.mean():.3f} ± DP {r2_blocos.std():.3f}")

    # O tutorial também faz o accuracy score, que eu deveria adequar para
    # as métricas, mas acho que acabaria dando na mesma.


# %% [markdown]
# # KNN Regressor

# %%
def knn_regressor(df, variaveis=None, n_blocos=5, embaralhar=False):
    """
    Roda um algoritmo de previsão de valores de uma determinada coluna.
    Roda esse algoritmo para todas as colunas de variáveis da base de dados.
    Avalia o modelo com validação cruzada: divide os meses em n_blocos, e cada bloco é testado
    uma vez, com o modelo treinado nos outros blocos.
    Em cada bloco, utiliza GridSearchCV para determinar a melhor parametrização do KNN,
    usando só os blocos de treino.
    Avalia o desempenho do modelo com as métricas de avaliação "MAE", "RMSE" e "R²",
    juntando as previsões de todos os blocos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse
    variaveis : list, optional
        Lista com nomes das colunas identificadas como variáveis de df, by default None
    n_blocos : int, optional
        Número de blocos da validação cruzada, by default 5
    embaralhar : bool, optional
        Se False, cada bloco é um período de meses seguidos (a base está em ordem de data),
        e o modelo é testado em anos que não viu. Se True, os meses são sorteados para os
        blocos (KFold aleatório, random_state=42), by default False

    Returns
    -------
    resultado : pd.DataFrame
        Uma linha por variável alvo, com as colunas:
        - variavel: variável prevista
        - rmse, mae, r2: métricas juntando as previsões de todos os blocos
          (cada mês é previsto uma vez, quando o bloco dele é o de teste)
        - r2_blocos: lista com o R² de cada bloco
        - melhores_parametros: parâmetros escolhidos pelo GridSearchCV no maior número de blocos
        - n_blocos_parametros: em quantos blocos esses parâmetros foram escolhidos
    """
    X = df.dropna().copy()

    if variaveis is None:
                from pcj.utils import separar_colunas
                _, variaveis = separar_colunas(df)

    # 1) Bibliotecas

    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

    # 2) Seleção de variável e normalização de colunas

    colunas = pd.Series(variaveis)

    normalize = MinMaxScaler(feature_range=(0,1))

    limpos_norm = normalize.fit_transform(X[variaveis])

    limpos_norm = pd.DataFrame(limpos_norm, columns=colunas)

    # 3) Validação cruzada e grade de hiperparâmetros

    # Divisão dos meses em blocos. Sem embaralhar, o KFold corta a base na ordem das linhas
    # (ordem de data): cada bloco é um período de meses seguidos
    if embaralhar:
        kfold = KFold(n_splits=n_blocos, shuffle=True, random_state=42)
    else:
        kfold = KFold(n_splits=n_blocos)

    # Ajuste dos Hiperparâmetros com GridSearchCV
    ## Definição da grade de hiperparâmetros
    param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15], # Número de vizinhos considerados pelo algoritmo
    'weights': ['uniform', 'distance'], # Como o algoritmo deve pesar as distâncias dos vizinhos
                                        # 'uniform' = uniformemente, 'distance' = em função da distância entre vizinhos
    'metric': ['euclidean', 'manhattan', 'minkowski']   # Que métrica de distância usar
    }

    # Lista que será convertida em DataFrame, com as métricas de todas as variáveis
    linhas = []

    # 4) Loop

    # Selecionará cada variável como variável alvo
    for target_col in colunas:

        y = limpos_norm[target_col]
        x = limpos_norm.drop(target_col, axis=1)    # Features = Todas as colunas, exceto a alvo

        # Previsão de cada mês, feita quando o bloco dele é o de teste
        y_pred = pd.Series(index=y.index, dtype=float)
        r2_blocos = []
        parametros_blocos = []

        # Cada volta do laço usa um bloco como teste e os outros como treino
        for treino, teste in kfold.split(x):
            x_train, x_test = x.iloc[treino], x.iloc[teste]
            y_train, y_test = y.iloc[treino], y.iloc[teste]

            # Executando o GridSearchCV só com os blocos de treino.
            # O cv=5 é uma segunda validação cruzada, dentro do treino, que escolhe os parâmetros
            grid_search = GridSearchCV(KNeighborsRegressor(), param_grid,
                                    cv=5)   # Coloquei 5 por causa do tutorial, não sei do que sse trata

            grid_search.fit(x_train, y_train)

            # Treinamento com os Melhores Parâmetros
            ## Seleção dos melhores hiperparâmetros identificados pelo GridSearchCV
            best_knn = KNeighborsRegressor(**grid_search.best_params_)

            ## Treinamento
            best_knn.fit(x_train, y_train)

            # Previsão do bloco de teste
            y_pred.iloc[teste] = best_knn.predict(x_test)

            r2_blocos.append(r2_score(y_test, y_pred.iloc[teste]))

            # str transforma o dicionário de parâmetros em texto, para contar quantas vezes ele se repete
            parametros_blocos.append(str(grid_search.best_params_))

        # Parâmetros escolhidos no maior número de blocos (value_counts conta e ordena do mais frequente)
        contagem = pd.Series(parametros_blocos).value_counts()

        # Avaliação do modelo com Métricas, juntando as previsões de todos os blocos
        linhas.append({
            "variavel": target_col,
            "rmse": root_mean_squared_error(y, y_pred),
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred),
            "r2_blocos": [round(r, 3) for r in r2_blocos],
            "melhores_parametros": contagem.index[0],
            "n_blocos_parametros": contagem.iloc[0]
        })

    return pd.DataFrame(linhas)


# %% [markdown]
# # K-Means - Clustering

# %%
def k_means(df, variaveis=None, n_clusters=10):
      """
      Roda o algoritmo de agrupamento K-Means.
      Roda 3 casos: 
      1. Dados brutos,
      2. Dados brutos pós normalização, 
      3. Dados normalizados pós redução de dimensionalidade PCA.
      Imprime as matrizes contendo todas as distâncias, rótulos e centros
      calculados pelo algoritmo em cada caso.

      Parameters
      ----------
      df : pd.DataFrame
          DataFrame de interesse.
      variaveis : list, optional
          Lista com nomes das colunas identificadas como variáveis de df, by default None
      n_clusters : int, optional
          Define o número máximo de clusters que será testado pelo modelo, by default 10

      Returns
      -------
      distancias : list
          Lista contendo matrizes com todas as distâncias calculadas pelo algoritmo.
          Uma matriz para cada caso: Dados Brutos, Dados Normalizados, Dados com Redução
          de Dimensionalidade PCA
      rotulos : list
          Lista contendo matrizes com todos os rótulos calculados pelo algoritmo.
          Uma matriz para cada caso: Dados Brutos, Dados Normalizados, Dados com Redução
          de Dimensionalidade PCA
      centros : list
          Lista contendo matrizes com todos os centros calculados pelo algoritmo.
          Uma matriz para cada caso: Dados Brutos, Dados Normalizados, Dados com Redução
          de Dimensionalidade PCA

      """      
      X = df.copy()

      if variaveis is None:
            from pcj.utils import separar_colunas
            _, variaveis = separar_colunas(df)

      # Bibliotecas

      import numpy as np
      from sklearn.cluster import KMeans
      from sklearn.preprocessing import StandardScaler
      from sklearn.decomposition import PCA

      # Eliminar células vazias
      X_nonan = X[variaveis].dropna(axis=0, how='any')

      # Salvar o Index do DataFrame sem células vazias numa variável
      idx_nonan = X_nonan.index

      # Normalização dos dados. É um pré requisito para o PCA funcionar
      scaler = StandardScaler()   # Coloca todas as variáveis na mesma escala, com média = 0 e desvio padrão = 1.
      X_scaled = scaler.fit_transform(X_nonan)

      # Redução da dimensionalidade
      pca = PCA(n_components=0.95, # Mantém 95% da variância
            random_state=0)
      X_pca = pca.fit_transform(X_scaled)

      # KMeans - "Bruto" (X_nonan)

      kmeans_raw = KMeans(n_clusters=2, # Número de clusters, ajustar para o que for preferido.
                        random_state=42)
      cluster_labels_raw = kmeans_raw.fit_predict(X_nonan)

      ## Distâncias euclidianas entre os pontos. Como não foi normalizado, há uma grande variabilidade entre as distâncias
      distancias_raw = np.linalg.norm(X_nonan - kmeans_raw.cluster_centers_[cluster_labels_raw], axis=1)

      # KMeans - Normalizado

      kmeans_norm = KMeans(n_clusters=2, 
                        random_state=42)
      cluster_labels_norm = kmeans_norm.fit_predict(X_scaled)

      ## Distâncias euclidianas bem menores, pós normalização
      distancias_norm = np.linalg.norm(X_scaled - kmeans_norm.cluster_centers_[cluster_labels_norm], axis=1)

      # KMeans - PCA

      kmeans_pca = KMeans(n_clusters=2,   
                  random_state=42)
      cluster_labels_pca = kmeans_pca.fit_predict(X_pca)

      ## Valores similares ao do normalizado, mas com a vantagem de estar em 2 dimensões, permitindo gerar visualizações. 
      ## A desvantagem é a perda de 5% de informação, devido ao `n_components = 0.95` de antes.
      distancias_pca = np.linalg.norm(X_pca - kmeans_pca.cluster_centers_[cluster_labels_pca], axis=1)

      distancias = [distancias_raw, distancias_norm, distancias_pca]
      rotulos = [kmeans_raw.labels_, kmeans_norm.labels_, kmeans_pca.labels_]
      centros = [kmeans_raw.cluster_centers_, kmeans_norm.cluster_centers_, kmeans_pca.cluster_centers_]

      print("Distâncias Brutas: " , 2*"\n" , distancias_raw , 2*"\n" , "Rótulos Brutos: " , 2*"\n" , rotulos[0] , 2*"\n" , "---------------------------------", "\n",
            "Distâncias Normalizadas: " , 2*"\n" , distancias_norm , 2*"\n" , "Rótulos Normalizados: " , 2*"\n" , rotulos[1] , 2*"\n" , "---------------------------------", "\n",
            "Distâncias PCA: " , 2*"\n" , distancias_pca , 2*"\n" , "Rótulos PCA: " , 2*"\n" , rotulos[2])

      return distancias, rotulos, centros

