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
    contaminacao_inicio=None,
    contaminacao_fim=None,
    contaminacao_passo=None,
    n_seeds=30,
    limiar=0.5,
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
    contaminacao_inicio : float, optional
        Valor mínimo de contaminação a ser testado, by default None
    contaminacao_fim : float, optional
        Valor máximo de contaminação (não incluído no intervalo), by default None
    contaminacao_passo : float, optional
        Intervalo entre os valores de contaminação, by default None
    n_seeds : int, optional
        Número de seeds (random_state de 0 a n_seeds - 1) por contaminação, by default 30
    limiar : float, optional
        Fração mínima de rodadas para considerar a linha uma anomalia por consenso, by default 0.5

    Returns
    -------
    frequencia : pd.Series
        Fração das rodadas em que cada linha foi marcada (0 a 1), com o índice de df.
    consenso : pd.Series of bool
        True para as linhas com frequencia >= limiar.

    Raises
    ------
    ValueError
        Se o usuário for parametrizar os valores de contaminação,
        ele deve inserir corretamente um valor mínimo, um máximo,
        e um passo.
    """
    # Bibliotecas
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest

    # Separação de variaveis
    if variaveis is None:
        from pcj.utils import separar_colunas
        _, variaveis = separar_colunas(df)

    # Intervalo de valores de contaminação a serem testados
    intervalo = (
        contaminacao_inicio,
        contaminacao_fim,
        contaminacao_passo,
    )

    if all(valor is None for valor in intervalo):
        valores_contaminacao = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

    elif any(valor is None for valor in intervalo):
        raise ValueError(
            "Para gerar o intervalo, informe início, fim e passo."
        )
    else:
        valores_contaminacao = np.arange(
            contaminacao_inicio,
            contaminacao_fim,
            contaminacao_passo,
        ).round(10).tolist()

    # Base de dados a ser utilizada
    X = df[variaveis].copy()

    # Dicionário onde as marcações de cada rodada serão armazenadas
    # Chave: (contaminação, seed). Valor: True/False para cada linha (True = anomalia)
    marcacoes = {}

    for contaminacao in valores_contaminacao:
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

    print(f"{len(valores_contaminacao)} contaminações x {n_seeds} seeds = {marcacoes.shape[1]} rodadas")
    print(f"Linhas marcadas em pelo menos {limiar:.0%} das rodadas: {consenso.sum()}")

    return frequencia, consenso



# %% [markdown]
# # Random Forest - Classificação

# %%
def random_forest(df, variaveis=None, var_alvo='pH'):
    """
    Roda um algoritmo de classificação com o modelo Random Forest.
    Testa a acurácia das classificações do modelo com as métricas 
    de avaliação "MAE", "RMSE" e "R²", e com uma validação cruzada.
    Imprime esses resultados no final.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse
    variaveis : list, optional
        Lista com nomes das colunas identificadas como variaveis de df, by default None
    var_alvo : str, optional
        Variável alvo de classificação, by default 'pH'
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

    # Divisão de treino e teste
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                        test_size=0.3,  # Estou copiando do tutorial
                                                        random_state=42
                                                        )
    
    # Criando o modelo
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=100,    # Copiando do tutorial
                                random_state=42)

    # Treinando o modelo
    model.fit(x_train, y_train)

    # Previsões
    y_pred = model.predict(x_test)

    # Avaliação do Desempenho
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

    print("RMSE:", root_mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("r²:", r2_score(y_test, y_pred))

    # Validação Cruzada
    from sklearn.model_selection import cross_val_score

    print("Validação Cruzada:", cross_val_score(model, x_test, y_test, cv=10))

    # O tutorial também faz o accuracy score, que eu deveria adequar para
    # as métricas, mas acho que acabaria dando na mesma.


# %% [markdown]
# # KNN Regressor

# %%
def knn_regressor(df, variaveis=None):
    """
    Roda um algoritmo de previsão de valores de uma determinada coluna.
    Roda esse algoritmo para todas as colunas de variáveis da base de dados.
    Utiliza GridSearchCV para determinar a melhor parametrização do KNN para cada variável.
    Avalia o desempenho do modelo com as métricas de avaliação "MAE", "RMSE" e "R²".
    Imprime os resultados de cada métrica, para cada variável no final.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse
    variaveis : list, optional
        Lista com nomes das colunas identificadas como variáveis de df, by default None
    """
    X = df.dropna().copy()

    if variaveis is None:
                from pcj.utils import separar_colunas
                _, variaveis = separar_colunas(df)

    # 1) Bibliotecas

    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

    # 2) Seleção de variável e normalização de colunas

    colunas = pd.Series(variaveis)

    normalize = MinMaxScaler(feature_range=(0,1))

    limpos_norm = normalize.fit_transform(X[variaveis])

    limpos_norm = pd.DataFrame(limpos_norm, columns=colunas)

    # 3) Loop

    # Selecionará cada variável como variável alvo
    for target_col in colunas:
        print(f"Variável: {target_col}", "\n")

        y = limpos_norm[target_col]
        x = limpos_norm.drop(target_col, axis=1)    # Features = Todas as colunas, exceto a alvo
        
        # Divisão em conjuntos de treinamento e teste para treinamento
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        # Inicialização do modelo
        model = KNeighborsRegressor(n_neighbors=5)

        # Treinamento
        model.fit(x_train, y_train)

        # Avaliação do Modelo com o método `score`
        result = model.score(x_test, y_test)
        print(f"Precisão do modelo no conjunto de teste: {result: .2%}")

        # Ajuste dos Hiperparâmetros com GridSearchCV
        ## Definição da grade de hiperparâmetros
        param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15], # Número de vizinhos considerados pelo algoritmo
        'weights': ['uniform', 'distance'], # Como o algoritmo deve pesar as distâncias dos vizinhos
                                            # 'uniform' = uniformemente, 'distance' = em função da distância entre vizinhos
        'metric': ['euclidean', 'manhattan', 'minkowski']   # Que métrica de distância usar
        }

        # Executando o GridSearchCV
        grid_search = GridSearchCV(KNeighborsRegressor(), param_grid,
                                cv=5)   # Coloquei 5 por causa do tutorial, não sei do que sse trata

        grid_search.fit(x_train, y_train)

        ## Resultados

        print("Melhores parâmetros: ", grid_search.best_params_)
        
        # Treinamento com os Melhores Parâmetros
        ## Seleção dos melhores hiperparâmetros identificados pelo GridSearchCV
        best_knn = KNeighborsRegressor(**grid_search.best_params_)

        ## Treinamento
        best_knn.fit(x_train, y_train)

        # Avaliação do modelo com Métricas
        y_pred = best_knn.predict(x_test)

        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        print("--------------")


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

