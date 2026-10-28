# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv (3.14.2.final.0)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# %% [markdown]
# # Funções auxiliares

# %%
def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


# %%
def evaluate_regressor(model, X_test, y_test, feature_names=None, target_name=None):
    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)
    return {
        'model': model,
        'x_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'metrics': metrics,
        'feature_names': feature_names,
        'target': target_name
    }


# %% [markdown]
# # Preparação para cálculos das métricas

# %%
def preparar_para_metricas(df_verdadeiro, df_predito, metadados, verbose=False):
    """
    Prepara dois DataFrames para cálculo de métricas.
    
    Remove metadados e garante que ambos tenham as mesmas colunas numéricas.
    
    Parameters
    ----------
    df_verdadeiro : pd.DataFrame
        DataFrame com valores verdadeiros.
    df_predito : pd.DataFrame
        DataFrame com valores preditos/imputados.
    metadados : list
        Lista de colunas de metadados a ignorar.
    verbose : bool, default False
        Se True, exibe (print) as colunas usadas nas métricas.
    
    Returns
    -------
    df_verdadeiro_prep : pd.DataFrame
        DataFrame preparado (apenas colunas numéricas comuns, sem metadados).
    df_predito_prep : pd.DataFrame
        DataFrame preparado (apenas colunas numéricas comuns, sem metadados).
    colunas_usadas : list
        Lista das colunas que serão usadas nas métricas.
    """
    # Eu presumo que os dados_verdadeiros serão uma das bases de dados produzidas até
    # o momento, e que os dados_preditos terão alguma coisa a ver com os modelos de ML.

    # Remove linhas com NaN
    df_verdadeiro = df_verdadeiro.dropna().copy()
    
    # Encontra colunas comuns (intersecção), na ordem das colunas de df_verdadeiro.
    # Não usa set: a ordem de um set muda a cada execução do Python, e as tabelas de resultado mudariam de ordem
    colunas_comuns = [col for col in df_verdadeiro.columns if col in df_predito.columns]
    
    # Remove metadados. Importante, pois os cálculos de métricas só funcionam em colunas numéricas.
    # Exemplo: Nas colunas de datas (dtype=datetime), os métodos sklearn retornam TypeError
    colunas_usadas = [col for col in colunas_comuns if col not in metadados]
    
    df_verdadeiro_prep = df_verdadeiro[colunas_usadas]
    df_predito_prep = df_predito[colunas_usadas]
    
    # Usa apenas as colunas que existem em ambos 
    colunas_usadas = list(df_verdadeiro_prep.columns)
    df_predito_prep = df_predito_prep[colunas_usadas]
    
    if verbose:
        print(f"Colunas usadas para métricas: {colunas_usadas}")
        print(f"Total de colunas: {len(colunas_usadas)}")
    
    return df_verdadeiro_prep, df_predito_prep, colunas_usadas


# %% [markdown]
# # Introdução artificial de NaN

# %%
def simular_dados_faltantes(df, pct_remover=0.05, seed=42, verbose=False):
    """
    Versão simplificada: cria dataset com NaN simulado.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset base (será removido NaN primeiro).
    pct_remover : float
        Percentual de valores a remover (ex: 0.05 para 5%).
    seed : int
        Seed para reprodutibilidade.
    verbose : bool, default False
        Se True, exibe (print) quantas colunas receberam NaN e a porcentagem removida.
    
    Returns
    -------
    df_sim : pd.DataFrame
        Dataset com NaN simulado com valores removidos aleatoriamente.
    mascara : pd.DataFrame
        Máscara marcando as coordenadas dos dados faltantes.
    """
    
    np.random.seed(seed)
    
    from pcj.utils import separar_colunas 
    metadados, variaveis = separar_colunas(df)
    
    # Remove todas as linhas com NaN
    df_ref = df.dropna().copy()
    
    # Todas as colunas numéricas (exceto metadados)
    todas_cols = [col for col in df_ref.columns if col not in metadados]
    
    # Cria dataset simulado
    df_sim = df_ref.copy()

    n_remover = int(len(df_sim) * pct_remover)

    # Cria a máscara com as coordenadas dos dados faltantes
    mascara = pd.DataFrame(False, index=df_ref.index, columns=df_ref.columns)
    
    # Remove valores aleatoriamente de TODAS as colunas (exceto metadados)
    for col in todas_cols:
        indices = np.random.choice(df_sim.index, n_remover, replace=False)
        df_sim.loc[indices, col] = np.nan
        mascara.loc[indices, col] = True
    
    if verbose:
        print(f"{len(todas_cols)} colunas, {pct_remover*100}% removido")

    return df_sim, mascara


# %% [markdown]
# # Métricas de Avaliação

# %% [markdown]
# Fluxograma simplificado:
# 0) Carregar dados
#
# 1) Usar a função simular_dados_faltantes: É preciso especificar o dataframe verdadeiro e a porcentagem de dados a serem removidos (default 5%, valor utilizado para imputação via média e mediana). (i) Ela remove linhas com NaN, (ii) introduz NaN artificiais, (iii) retorna um dataframe com NaN artificiais em todas as colunas, e (iv) a máscara com as coordenadas desses NaN.
#
# 2) Fazer a imputação no dataframe artificial gerado na etapa anterior. É preciso especificar qual DataFrame, e se deseja-se que o DataFrame pós imputação apresente apenas as colunas que foram manipuladas (return_reduced=True)
#
# 3) Fazer avaliação de imputação, utilizando a função avaliar_imputacao: É preciso especificar: (i) o dataframe real, (ii) o dataframe imputado, (iii) a máscara do dataframe imputado, e (iv) o nome do método de imputação.

# %%
def avaliar_imputacao(df_verdadeiro, df_imputado, mascara,
                        metodo, pct_faltantes=None, verbose=False, devolver_celulas=False):
    """
    Interface simples para avaliar qualidade de imputação.
    
    Compara dataset verdadeiro (referência) com dataset imputado, calculando
    métricas de desempenho (RMSE, MAE, R²) para cada variável.
    
    Parameters
    ----------
    df_verdadeiro : pd.DataFrame
        Dataset verdadeiro (sem manipulações).
    df_imputado : pd.DataFrame
        Dataset que recebeu alguma transformação (imputação, etc).
    mascara : pd.DataFrame
        Máscara booleana marcando as coordenadas de dados faltantes
        (True onde há valores imputados).
    metodo : str
        Nome do método de imputação sendo avaliado.
        Valores esperados: "media", "mediana" ou "knn".
    pct_faltantes : float, optional
        Porcentagem de dados faltantes (ex: 0.05 para 5%).
        Padrão: None
    verbose : bool, default False
        Se True, exibe (print) as colunas usadas nas métricas.
    devolver_celulas : bool, default False
        Se True, devolve também a tabela de células (valor verdadeiro e imputado de cada célula).
    
    Returns
    -------
    resultado : pd.DataFrame
        DataFrame com métricas de avaliação por variável, contendo colunas:
        - variavel: nome da coluna avaliada
        - rmse: erro quadrático médio
        - mae: erro absoluto médio
        - r2: coeficiente de determinação
        - n_celulas: número de células avaliadas
        - metodo: método de imputação
        - pct_faltantes: percentual de dados faltantes
    celulas : pd.DataFrame, optional
        Apenas se devolver_celulas=True. Uma linha por célula avaliada, com as colunas
        variavel, indice (rótulo da linha no DataFrame), verdadeiro, imputado, metodo e pct_faltantes.
    """
    
    from pcj.utils import separar_colunas
    metadados, variaveis = separar_colunas(df_verdadeiro)
    
    # Preparar dados
    df_verd_prep, df_imput_prep, colunas = preparar_para_metricas(
        df_verdadeiro, df_imputado, metadados, verbose=verbose
    )
    
    # Alinhar máscara ao mesmo índice e colunas usadas na métrica
    mascara_prep = mascara.loc[df_verd_prep.index, colunas]

    # Cria uma lista que será convertida em DataFrame, com todas as métricas de todas as variáveis

    linhas = []

    # Lista de tabelas de células (uma por variável), usada só se devolver_celulas=True
    celulas = []

    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

    for col in colunas:
        m = mascara_prep[col]
        if not m.any():
            continue
        y_true = df_verd_prep.loc[m, col]
        y_pred = df_imput_prep.loc[m, col]

        # Passa um dicionário para a lista "linhas", onde cada métrica de cada variável será armazenada numa linha diferente
        linhas.append({
            "variavel": col,
            "rmse": root_mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "n_celulas": len(y_true),
            "metodo": metodo,
            "pct_faltantes": pct_faltantes
        })

        # Guarda as células desta variável, para calcular as métricas juntando várias rodadas
        if devolver_celulas:
            celulas.append(pd.DataFrame({
                "variavel": col,
                "indice": y_true.index,
                "verdadeiro": y_true.values,
                "imputado": y_pred.values,
                "metodo": metodo,
                "pct_faltantes": pct_faltantes
            }))
    
    # Converte a lista "linhas" para DataFrame
    resultado = pd.DataFrame(linhas)

    if devolver_celulas:
        # pd.concat não aceita uma lista vazia: sem células avaliadas, devolve uma tabela vazia
        if celulas:
            celulas = pd.concat(celulas, ignore_index=True)
        else:
            celulas = pd.DataFrame()
        return resultado, celulas

    return resultado


# %%
def avaliacao_completa(df, n_seeds=30, devolver_celulas=False):
    """
    Orquestra a avaliação completa de todos os métodos de imputação.
    
    Executa um pipeline completo que simula dados faltantes, aplica os três
    métodos de imputação (média, mediana, KNN), e calcula
    as métricas de desempenho para cada combinação.
    Repete tudo n_seeds vezes: em cada rodada, a seed sorteia outras células para remover.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse. As linhas com NaN são descartadas; as linhas completas
        que sobram são a referência (valores verdadeiros).
    n_seeds : int, default 30
        Número de rodadas. As seeds vão de 0 a n_seeds - 1.
    devolver_celulas : bool, default False
        Se True, devolve também a tabela de células de todas as rodadas,
        para calcular as métricas juntando as rodadas (função metricas_juntando).
    
    Returns
    -------
    tabela_completa : pd.DataFrame
        DataFrame consolidado, com uma linha por seed × método × variável:
        - média e mediana: 5% de dados removidos;
        - KNN: 30% de dados removidos.
        
        Colunas incluem: variavel, rmse, mae, r2, n_celulas, metodo, pct_faltantes, seed.
    celulas : pd.DataFrame, optional
        Apenas se devolver_celulas=True. Uma linha por célula avaliada em cada rodada:
        variavel, indice, verdadeiro, imputado, metodo, pct_faltantes, seed.
    
    Notes
    -----
    - Método média/mediana usa 5% de remoção de dados;
    - Método KNN usa 30% de remoção de dados;
    - Resultados são concatenados em um único DataFrame.
    """

    from pcj.imputacao import mean_imput, median_imput, knn_imput
    
    ref = df.dropna().copy()

    partes = []
    celulas = []

    # Cada seed sorteia outras células para remover. Repetir aumenta a amostra:
    # com 5% de remoção, cada rodada avalia poucas células por variável
    for seed in range(n_seeds):

        # Para média e mediana
        sim, mascara = simular_dados_faltantes(
            df, pct_remover=0.05, seed=seed
        )

        metodos = [("media", mean_imput), ("mediana", median_imput)]

        for nome, func in metodos:
            imputado = func(sim, return_reduced=True)
            resultado, cel = avaliar_imputacao(ref, imputado, mascara, metodo=nome, pct_faltantes=0.05,
                                               devolver_celulas=True)
            resultado["seed"] = seed    # Identifica a rodada na tabela final
            cel["seed"] = seed
            partes.append(resultado)
            celulas.append(cel)

        # Para KNN (mesmo loop de antes, só que com ajustes para acomodar o KNN)
        sim, mascara = simular_dados_faltantes(
            df, pct_remover=0.3, seed=seed
        )

        metodos = [("knn", knn_imput)]

        for nome, func in metodos:
            imputado = func(sim, return_reduced=True)
            resultado, cel = avaliar_imputacao(ref, imputado, mascara, metodo=nome, pct_faltantes=0.3,
                                               devolver_celulas=True)
            resultado["seed"] = seed
            cel["seed"] = seed
            partes.append(resultado)
            celulas.append(cel)
    
    tabela_completa = pd.concat(partes, ignore_index=True)

    if devolver_celulas:
        return tabela_completa, pd.concat(celulas, ignore_index=True)

    return tabela_completa


# %%
def metricas_juntando(celulas):
    """
    Calcula RMSE, MAE e R² juntando as células de todas as rodadas (seeds).
    
    Em vez de calcular uma métrica por rodada e tirar a média, junta todas as células
    de um método e de uma variável e calcula cada métrica uma vez só. Com poucas células
    por rodada, o R² de uma rodada oscila muito; juntando, ele fica estável.
    
    Parameters
    ----------
    celulas : pd.DataFrame
        Tabela de células devolvida por avaliacao_completa(..., devolver_celulas=True),
        com as colunas variavel, verdadeiro, imputado, metodo, pct_faltantes e seed.
    
    Returns
    -------
    resultado : pd.DataFrame
        Uma linha por método × variável, com as colunas variavel, rmse, mae, r2,
        n_celulas (total de células juntadas), n_seeds, metodo e pct_faltantes.
    """
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

    linhas = []

    # Cada grupo = todas as células de um método e de uma variável, de todas as rodadas
    for (metodo, variavel), grupo in celulas.groupby(["metodo", "variavel"], sort=False):
        y_true = grupo["verdadeiro"]
        y_pred = grupo["imputado"]

        linhas.append({
            "variavel": variavel,
            "rmse": root_mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "n_celulas": len(grupo),
            "n_seeds": grupo["seed"].nunique(),
            "metodo": metodo,
            "pct_faltantes": grupo["pct_faltantes"].iloc[0]
        })

    return pd.DataFrame(linhas)
