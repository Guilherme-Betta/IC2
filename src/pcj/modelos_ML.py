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
def random_forest(df, variaveis=None, var_alvo='pH', n_blocos=5, embaralhar=False, seed_embaralhar=42,
                  meses_avaliados=None, devolver_previsoes=False, coluna_data="Data", verbose=False):
    """
    Roda um algoritmo de regressão com o modelo Random Forest.
    Avalia o modelo com validação cruzada: divide os meses em n_blocos, e cada bloco é testado
    uma vez, com o modelo treinado nos outros blocos.
    Testa a qualidade das previsões do modelo com as métricas de avaliação "MAE", "RMSE" e "R²",
    juntando as previsões de todos os blocos, e com o R² de cada bloco.
    Devolve esses resultados numa tabela (com verbose=True, também os imprime).

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
        blocos (KFold aleatório, com a seed seed_embaralhar), by default False
    seed_embaralhar : int, optional
        Seed do sorteio dos meses para os blocos (random_state do KFold). Só é usada com
        embaralhar=True; seeds diferentes dão divisões diferentes, by default 42
    meses_avaliados : list, optional
        Rótulos (índice de df) dos meses completos que são testados e entram nas métricas.
        Os outros meses completos entram no treino de todos os blocos, mas nunca são
        previstos. Serve para comparar duas bases com a mesma prova: os blocos são cortados
        só nos meses avaliados, na ordem das linhas. Se None, todos os meses completos são
        avaliados, by default None
    devolver_previsoes : bool, optional
        Se True, devolve também a previsão de cada mês avaliado, by default False
    coluna_data : str, optional
        Nome da coluna de df com a data de cada mês (usada na tabela de previsões),
        by default "Data"
    verbose : bool, optional
        Se True, imprime as métricas e o R² de cada bloco, by default False

    Returns
    -------
    resultado : pd.DataFrame
        Uma linha, com as colunas:
        - variavel: variável prevista (var_alvo)
        - rmse, mae, r2: métricas juntando as previsões de todos os blocos
          (cada mês avaliado é previsto uma vez, quando o bloco dele é o de teste)
        - r2_blocos: lista com o R² de cada bloco
    previsoes : pd.DataFrame, optional
        Apenas se devolver_previsoes=True. Uma linha por mês avaliado, com as colunas
        <coluna_data>, variavel, bloco (1 a n_blocos), verdadeiro e previsto.

    Raises
    ------
    ValueError
        Se algum mês de meses_avaliados não for um mês completo (sem NaN) de df.
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
        kfold = KFold(n_splits=n_blocos, shuffle=True, random_state=seed_embaralhar)
    else:
        kfold = KFold(n_splits=n_blocos)

    # Bibliotecas do modelo e das métricas
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

    # Meses avaliados (testados e medidos) e meses extras (sempre no treino), pela posição em X_nonan
    if meses_avaliados is None:
        avaliados = list(range(len(X_nonan)))
        extras = []
    else:
        # Um mês com NaN não serve: ele nem está em X_nonan
        if not pd.Index(meses_avaliados).isin(X_nonan.index).all():
            raise ValueError("meses_avaliados só pode ter meses completos (sem NaN) de df.")
        eh_avaliado = X_nonan.index.isin(meses_avaliados)
        avaliados = [p for p in range(len(X_nonan)) if eh_avaliado[p]]
        extras = [p for p in range(len(X_nonan)) if not eh_avaliado[p]]

    # Previsão de cada mês, feita quando o bloco dele é o de teste
    y_pred = pd.Series(index=y.index, dtype=float)
    r2_blocos = []
    linhas_previsoes = []
    bloco = 0

    # Cada volta do laço usa um bloco como teste e os outros como treino.
    # Os blocos são cortados só nos meses avaliados: o KFold devolve posições na lista avaliados,
    # que são convertidas em posições de x. Os meses extras entram sempre no treino
    for treino, teste in kfold.split(avaliados):
        bloco = bloco + 1
        pos_teste = [avaliados[i] for i in teste]
        pos_treino = sorted([avaliados[i] for i in treino] + extras)

        x_train, x_test = x.iloc[pos_treino], x.iloc[pos_teste]
        y_train, y_test = y.iloc[pos_treino], y.iloc[pos_teste]

        # Criando o modelo
        model = RandomForestRegressor(n_estimators=100,    # Copiando do tutorial
                                    random_state=42)

        # Treinando o modelo
        model.fit(x_train, y_train)

        # Previsões do bloco de teste
        y_pred.iloc[pos_teste] = model.predict(x_test)

        r2_blocos.append(r2_score(y_test, y_pred.iloc[pos_teste]))

        # Previsão de cada mês do bloco, para a tabela de previsões
        if devolver_previsoes:
            for p in pos_teste:
                linhas_previsoes.append({
                    coluna_data: X_nonan[coluna_data].iloc[p],
                    "variavel": var_alvo,
                    "bloco": bloco,
                    "verdadeiro": y.iloc[p],
                    "previsto": y_pred.iloc[p]
                })

    # Avaliação do Desempenho, juntando as previsões de todos os blocos (só os meses avaliados)
    y_aval = y.iloc[avaliados]
    pred_aval = y_pred.iloc[avaliados]
    rmse = root_mean_squared_error(y_aval, pred_aval)
    mae = mean_absolute_error(y_aval, pred_aval)
    r2 = r2_score(y_aval, pred_aval)

    # Validação Cruzada: R² de cada bloco e média ± DP dos blocos
    r2_blocos = pd.Series(r2_blocos)

    if verbose:
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("r²:", r2)
        print("R² de cada bloco:", r2_blocos.round(3).tolist())
        print(f"R² dos blocos: média {r2_blocos.mean():.3f} ± DP {r2_blocos.std():.3f}")

    # Tabela com uma linha, no mesmo formato da do knn_regressor
    resultado = pd.DataFrame([{
        "variavel": var_alvo,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "r2_blocos": r2_blocos.round(3).tolist()
    }])

    # O tutorial também faz o accuracy score, que eu deveria adequar para
    # as métricas, mas acho que acabaria dando na mesma.

    if devolver_previsoes:
        return resultado, pd.DataFrame(linhas_previsoes)

    return resultado


# %% [markdown]
# # KNN Regressor

# %%
def knn_regressor(df, variaveis=None, n_blocos=5, embaralhar=False, seed_embaralhar=42,
                  meses_avaliados=None, devolver_previsoes=False, coluna_data="Data"):
    """
    Roda um algoritmo de previsão de valores de uma determinada coluna.
    Roda esse algoritmo para todas as colunas de variáveis da base de dados.
    Avalia o modelo com validação cruzada: divide os meses em n_blocos, e cada bloco é testado
    uma vez, com o modelo treinado nos outros blocos.
    Em cada bloco, utiliza GridSearchCV para determinar a melhor parametrização do KNN,
    usando só os blocos de treino.
    Avalia o desempenho do modelo com as métricas de avaliação "MAE", "RMSE" e "R²",
    juntando as previsões de todos os blocos. O modelo trabalha com as variáveis normalizadas
    (0 a 1); antes do RMSE e do MAE, as previsões voltam para a unidade original da variável.

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
        blocos (KFold aleatório, com a seed seed_embaralhar), by default False
    seed_embaralhar : int, optional
        Seed do sorteio dos meses para os blocos (random_state do KFold). Só é usada com
        embaralhar=True; seeds diferentes dão divisões diferentes, by default 42
    meses_avaliados : list, optional
        Rótulos (índice de df) dos meses completos que são testados e entram nas métricas.
        Os outros meses completos entram no treino de todos os blocos, mas nunca são
        previstos. Serve para comparar duas bases com a mesma prova: os blocos são cortados
        só nos meses avaliados, na ordem das linhas. Se None, todos os meses completos são
        avaliados, by default None
    devolver_previsoes : bool, optional
        Se True, devolve também a previsão de cada mês avaliado, by default False
    coluna_data : str, optional
        Nome da coluna de df com a data de cada mês (usada na tabela de previsões),
        by default "Data"

    Returns
    -------
    resultado : pd.DataFrame
        Uma linha por variável alvo, com as colunas:
        - variavel: variável prevista
        - rmse, mae, r2: métricas juntando as previsões de todos os blocos
          (cada mês avaliado é previsto uma vez, quando o bloco dele é o de teste).
          RMSE e MAE na unidade original da variável; o R² não tem unidade
        - r2_blocos: lista com o R² de cada bloco
        - melhores_parametros: parâmetros escolhidos pelo GridSearchCV no maior número de blocos
        - n_blocos_parametros: em quantos blocos esses parâmetros foram escolhidos
    previsoes : pd.DataFrame, optional
        Apenas se devolver_previsoes=True. Uma linha por mês avaliado e variável alvo, com
        as colunas <coluna_data>, variavel, bloco (1 a n_blocos), verdadeiro e previsto
        (na unidade original).

    Raises
    ------
    ValueError
        Se algum mês de meses_avaliados não for um mês completo (sem NaN) de df.
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

    # Mínimo e amplitude (máximo - mínimo) de cada variável, os mesmos que o MinMaxScaler usou.
    # Servem para voltar as previsões à unidade original: valor = normalizado × amplitude + mínimo
    minimo = X[variaveis].min()
    amplitude = X[variaveis].max() - X[variaveis].min()

    # Meses avaliados (testados e medidos) e meses extras (sempre no treino), pela posição em X
    # (limpos_norm tem as mesmas linhas, na mesma ordem)
    if meses_avaliados is None:
        avaliados = list(range(len(X)))
        extras = []
    else:
        # Um mês com NaN não serve: ele nem está em X
        if not pd.Index(meses_avaliados).isin(X.index).all():
            raise ValueError("meses_avaliados só pode ter meses completos (sem NaN) de df.")
        eh_avaliado = X.index.isin(meses_avaliados)
        avaliados = [p for p in range(len(X)) if eh_avaliado[p]]
        extras = [p for p in range(len(X)) if not eh_avaliado[p]]

    # 3) Validação cruzada e grade de hiperparâmetros

    # Divisão dos meses em blocos. Sem embaralhar, o KFold corta a base na ordem das linhas
    # (ordem de data): cada bloco é um período de meses seguidos
    if embaralhar:
        kfold = KFold(n_splits=n_blocos, shuffle=True, random_state=seed_embaralhar)
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

    # Previsão de cada mês avaliado, de todas as variáveis (usada só se devolver_previsoes=True)
    linhas_previsoes = []

    # 4) Loop

    # Selecionará cada variável como variável alvo
    for target_col in colunas:

        y = limpos_norm[target_col]
        x = limpos_norm.drop(target_col, axis=1)    # Features = Todas as colunas, exceto a alvo

        # Previsão de cada mês, feita quando o bloco dele é o de teste
        y_pred = pd.Series(index=y.index, dtype=float)
        r2_blocos = []
        parametros_blocos = []
        bloco = 0

        # Cada volta do laço usa um bloco como teste e os outros como treino.
        # Os blocos são cortados só nos meses avaliados: o KFold devolve posições na lista avaliados,
        # que são convertidas em posições de x. Os meses extras entram sempre no treino
        for treino, teste in kfold.split(avaliados):
            bloco = bloco + 1
            pos_teste = [avaliados[i] for i in teste]
            pos_treino = sorted([avaliados[i] for i in treino] + extras)

            x_train, x_test = x.iloc[pos_treino], x.iloc[pos_teste]
            y_train, y_test = y.iloc[pos_treino], y.iloc[pos_teste]

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
            y_pred.iloc[pos_teste] = best_knn.predict(x_test)

            r2_blocos.append(r2_score(y_test, y_pred.iloc[pos_teste]))

            # Previsão de cada mês do bloco, na unidade original, para a tabela de previsões
            if devolver_previsoes:
                for p in pos_teste:
                    linhas_previsoes.append({
                        coluna_data: X[coluna_data].iloc[p],
                        "variavel": target_col,
                        "bloco": bloco,
                        "verdadeiro": X[target_col].iloc[p],
                        "previsto": y_pred.iloc[p] * amplitude[target_col] + minimo[target_col]
                    })

            # str transforma o dicionário de parâmetros em texto, para contar quantas vezes ele se repete
            parametros_blocos.append(str(grid_search.best_params_))

        # Parâmetros escolhidos no maior número de blocos (value_counts conta e ordena do mais frequente)
        contagem = pd.Series(parametros_blocos).value_counts()

        # Volta à unidade original, só nos meses avaliados: valor = normalizado × amplitude + mínimo.
        # Os valores verdadeiros vêm direto de X, que já está na unidade original
        y_aval = X[target_col].iloc[avaliados]
        pred_aval = y_pred.iloc[avaliados] * amplitude[target_col] + minimo[target_col]

        # Avaliação do modelo com Métricas, juntando as previsões de todos os blocos
        linhas.append({
            "variavel": target_col,
            "rmse": root_mean_squared_error(y_aval, pred_aval),
            "mae": mean_absolute_error(y_aval, pred_aval),
            "r2": r2_score(y_aval, pred_aval),
            "r2_blocos": [round(r, 3) for r in r2_blocos],
            "melhores_parametros": contagem.index[0],
            "n_blocos_parametros": contagem.iloc[0]
        })

    if devolver_previsoes:
        return pd.DataFrame(linhas), pd.DataFrame(linhas_previsoes)

    return pd.DataFrame(linhas)


# %% [markdown]
# # K-Means - Escolha do número de grupos (k)

# %%
def selecao_k(df, variaveis=None, k_min=2, k_max=10, n_seeds=30, criterio="cotovelo", verbose=False):
    """
    Testa quantidades de grupos (k) para o K-Means com três critérios: a inércia
    (método do cotovelo), o coeficiente de silhueta e o índice de Calinski-Harabasz.
    Usa o mesmo espaço da detecção de outliers (detect_outliers_pca_kmeans): os meses
    completos, padronizados e reduzidos por PCA (95% da variância).
    Cada k roda com n_seeds seeds, porque a posição inicial dos centros pode mudar os grupos.
    O critério escolhido decide o k (por padrão, o cotovelo, como no projeto, seção 3.4);
    os outros dois validam a escolha.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse.
    variaveis : list, optional
        Lista com nomes das colunas identificadas como variáveis de df. Se None, usa todas
        as colunas de df que não são metadados (separar_colunas), by default None
    k_min : int, optional
        Menor quantidade de grupos testada, by default 2
    k_max : int, optional
        Maior quantidade de grupos testada, by default 10
    n_seeds : int, optional
        Número de seeds do K-Means (random_state de 0 a n_seeds - 1), by default 30
    criterio : str, optional
        Critério que decide o k: "cotovelo", "silhueta" ou "calinski_harabasz".
        Se o kneed não achar cotovelo, a silhueta decide (com aviso), by default "cotovelo"
    verbose : bool, optional
        Rodar a função exibe (print) o k apontado por cada critério, by default False

    Returns
    -------
    rodadas : pd.DataFrame
        Uma linha por rodada (k, seed), com a inércia, a silhueta e o Calinski-Harabasz.
    resumo : pd.DataFrame
        Uma linha por k, com a média e o desvio padrão de cada critério entre as seeds,
        e três colunas True/False que marcam o k apontado por cada critério
        (indicado_silhueta, indicado_cotovelo, indicado_calinski_harabasz).
        Se o kneed não achar cotovelo, indicado_cotovelo fica False em todas as linhas.
    k_escolhido : int
        O k apontado pelo critério escolhido (com criterio="cotovelo" e sem cotovelo,
        o k da silhueta).
    """
    # Bibliotecas
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from kneed import KneeLocator

    if variaveis is None:
        from pcj.utils import separar_colunas
        _, variaveis = separar_colunas(df)

    # Confere o critério antes das rodadas, para um nome errado não esperar todas elas para dar erro
    criterios_validos = ["cotovelo", "silhueta", "calinski_harabasz"]

    if criterio not in criterios_validos:
        raise ValueError(f"criterio deve ser um destes: {criterios_validos}. Recebido: '{criterio}'")

    # 1. O PCA e o K-Means não aceitam NaN: só os meses completos entram
    X_nonan = df[variaveis].dropna(axis=0, how='any')

    # 2. Padronização: sem ela, as variáveis de valores grandes dominariam as distâncias
    X_scaled = StandardScaler().fit_transform(X_nonan)

    # 3. Componentes principais que explicam 95% da variância (como na detecção de outliers)
    X_pca = PCA(n_components=0.95).fit_transform(X_scaled)

    # 4. Cada k roda com cada seed, e cada rodada vira uma linha da tabela
    linhas = []

    for k in range(k_min, k_max + 1):
        for seed in range(n_seeds):
            kmeans = KMeans(n_clusters=k, random_state=seed)
            grupos = kmeans.fit_predict(X_pca)

            linhas.append({
                "k": k,
                "seed": seed,
                # Inércia: soma dos quadrados das distâncias de cada mês ao centro do seu grupo (cai sempre que k aumenta)
                "inercia": kmeans.inertia_,
                # Silhueta: de -1 a 1. Quanto maior, mais cada mês está perto do próprio grupo e longe do grupo vizinho
                "silhueta": silhouette_score(X_pca, grupos),
                # Calinski-Harabasz: dispersão entre os grupos ÷ dispersão dentro deles. Quanto maior, melhor
                "calinski_harabasz": calinski_harabasz_score(X_pca, grupos),
            })

    rodadas = pd.DataFrame(linhas)

    # 5. Resumo: média e desvio padrão de cada critério entre as seeds, para cada k.
    # O .std() do pandas usa ddof=1 (desvio padrão da amostra), como no experimento A
    linhas = []

    for k in range(k_min, k_max + 1):
        rodadas_k = rodadas[rodadas["k"] == k]

        linhas.append({
            "k": k,
            "inercia_media": rodadas_k["inercia"].mean(),
            "inercia_dp": rodadas_k["inercia"].std(),
            "silhueta_media": rodadas_k["silhueta"].mean(),
            "silhueta_dp": rodadas_k["silhueta"].std(),
            "calinski_harabasz_media": rodadas_k["calinski_harabasz"].mean(),
            "calinski_harabasz_dp": rodadas_k["calinski_harabasz"].std(),
        })

    resumo = pd.DataFrame(linhas)

    # 6. k apontado por cada critério
    # Silhueta e Calinski-Harabasz: o k com a maior média
    k_silhueta = int(resumo.loc[resumo["silhueta_media"].idxmax(), "k"])
    k_calinski = int(resumo.loc[resumo["calinski_harabasz_media"].idxmax(), "k"])

    # Cotovelo: o k em que a inércia média para de cair rápido (algoritmo Kneedle, biblioteca kneed).
    # A curva da inércia é convexa e decrescente. Se não houver cotovelo, o kneed devolve None
    cotovelo = KneeLocator(resumo["k"], resumo["inercia_media"], curve="convex", direction="decreasing")
    k_cotovelo = cotovelo.knee

    if k_cotovelo is not None:
        k_cotovelo = int(k_cotovelo)

    # Colunas True/False: True só na linha do k apontado (com k_cotovelo = None, a comparação dá False em todas)
    resumo["indicado_silhueta"] = resumo["k"] == k_silhueta
    resumo["indicado_cotovelo"] = resumo["k"] == k_cotovelo
    resumo["indicado_calinski_harabasz"] = resumo["k"] == k_calinski

    # 7. O critério escolhido decide (por padrão, o cotovelo, como no projeto, 3.4); os outros dois validam
    if criterio == "cotovelo":
        k_escolhido = k_cotovelo

        # Sem cotovelo, a silhueta decide. O aviso sai mesmo com verbose=False: a regra mudou
        if k_escolhido is None:
            k_escolhido = k_silhueta
            print(f"Aviso: o kneed não achou cotovelo na inércia; a silhueta decide (k = {k_silhueta})")
    elif criterio == "silhueta":
        k_escolhido = k_silhueta
    else:
        k_escolhido = k_calinski

    if verbose:
        print(f"K-Means em {len(X_nonan)} meses completos ({len(df) - len(X_nonan)} com NaN, não avaliados); "
              f"{X_pca.shape[1]} componentes principais")
        print(f"k de {k_min} a {k_max} x {n_seeds} seeds = {len(rodadas)} rodadas")
        print(f"Silhueta (maior média): k = {k_silhueta}")

        if k_cotovelo is None:
            print("Cotovelo da inércia (kneed): sem cotovelo")
        else:
            print(f"Cotovelo da inércia (kneed): k = {k_cotovelo}")

        print(f"Calinski-Harabasz (maior média): k = {k_calinski}")
        print(f"k escolhido (critério: {criterio}): {k_escolhido}")

    return rodadas, resumo, k_escolhido


# %% [markdown]
# # K-Means - Clustering

# %%
def k_means(df, variaveis=None, n_clusters=4, coluna_data="Data", verbose=False):
    """
    Agrupa os meses com o algoritmo K-Means, no mesmo espaço da detecção de outliers
    e da selecao_k: meses completos, padronizados e reduzidos por PCA (95% da variância).
    O K-Means testa 10 posições iniciais dos centros e fica com a de menor inércia
    (n_init=10, random_state=42), para o resultado não depender de um único sorteio.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse.
    variaveis : list, optional
        Lista com nomes das colunas identificadas como variaveis de df, by default None
    n_clusters : int, optional
        Número de grupos (k). O padrão é o k do cotovelo nestas bases; ver selecao_k
        para a escolha, by default 4
    coluna_data : str, optional
        Nome da coluna de df com a data de cada mês, by default "Data"
    verbose : bool, optional
        Rodar a função exibe (print) quantos meses ficaram em cada grupo, by default False

    Returns
    -------
    tabela : pd.DataFrame
        Uma linha por mês (índice = df.index), com a data e o grupo do mês.
        Os meses com dado faltante não são agrupados e ficam com grupo <NA>.
    perfil : pd.DataFrame
        Uma linha por grupo, com o número de meses e a média de cada variável na
        unidade original de df. O número do grupo é só um nome: o grupo 0 não é
        "menor" que o grupo 1.
    """
    # Bibliotecas
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    if variaveis is None:
        from pcj.utils import separar_colunas
        _, variaveis = separar_colunas(df)

    # Eliminar células vazias
    X_nonan = df[variaveis].dropna(axis=0, how='any')

    # Salvar o Index do DataFrame sem células vazias numa variável
    idx_nonan = X_nonan.index

    # Normalização dos dados. É um pré requisito para o PCA funcionar
    scaler = StandardScaler()   # Coloca todas as variáveis na mesma escala, com média = 0 e desvio padrão = 1.
    X_scaled = scaler.fit_transform(X_nonan)

    # Redução da dimensionalidade
    pca = PCA(n_components=0.95, # Mantém 95% da variância
          random_state=0)
    X_pca = pca.fit_transform(X_scaled)

    # KMeans - PCA: o melhor de 10 inícios (o de menor inércia), com seed fixa para ser reproduzível
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    grupos = kmeans.fit_predict(X_pca)

    # Tabela: uma linha por mês de df. O reindex estende para todos os meses: os não agrupados ficam NaN.
    # "Int64" (com I maiúsculo) é o inteiro do pandas que aceita faltantes: mostra 0 e 1 em vez de 0.0 e 1.0
    tabela = df[[coluna_data]].copy()
    tabela["grupo"] = pd.Series(grupos, index=idx_nonan).reindex(df.index).astype("Int64")

    # Perfil: uma linha por grupo, com a média de cada variável na unidade original
    linhas = []

    for grupo in range(n_clusters):
        meses_grupo = X_nonan[grupos == grupo]

        linha = {"grupo": grupo, "meses": len(meses_grupo)}

        for variavel in variaveis:
            linha[variavel] = meses_grupo[variavel].mean()

        linhas.append(linha)

    perfil = pd.DataFrame(linhas)

    if verbose:
        print(f"K-Means com k = {n_clusters} em {len(idx_nonan)} meses completos "
              f"({len(df) - len(idx_nonan)} com NaN, não agrupados); {X_pca.shape[1]} componentes principais")
        print("Meses por grupo:", perfil["meses"].tolist())

    return tabela, perfil

