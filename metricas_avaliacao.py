# %%
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from utils import dados, pivot, salvar

import numpy as np
import pandas as pd
import sklearn

# %% [markdown]
# # Preparação para cálculos das métricas

# %%
def preparar_para_metricas(df_verdadeiro, df_predito, metadados):
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
    
    # Encontra colunas comuns (intersecção)
    colunas_comuns = set(df_verdadeiro.columns) & set(df_predito.columns)
    
    # Remove metadados. Importante, pois os cálculos de métricas só funcionam em colunas numéricas.
    # Exemplo: Nas colunas de datas (dtype=datetime), os métodos sklearn retornam TypeError
    colunas_usadas = [col for col in colunas_comuns if col not in metadados]
    
    df_verdadeiro_prep = df_verdadeiro[colunas_usadas]
    df_predito_prep = df_predito[colunas_usadas]
    
    # Usa apenas as colunas que existem em ambos 
    colunas_usadas = list(df_verdadeiro_prep.columns)
    df_predito_prep = df_predito_prep[colunas_usadas]
    
    print(f"Colunas usadas para métricas: {colunas_usadas}")
    print(f"Total de colunas: {len(colunas_usadas)}")
    
    return df_verdadeiro_prep, df_predito_prep, colunas_usadas

# %% [markdown]
# # Introdução artificial de NaN

# %%
def simular_dados_faltantes(df, pct_remover=0.05, seed=42):
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
    
    Returns
    -------
    df_sim_a : pd.DataFrame
        Dataset com NaN simulado. Cenário "a".
        Cenário "a": NaN introduzidos apenas em colunas que tinham NaN originalmente.
    mascara_a : pd.DataFrame
        Máscara marcando as coordenadas de dados faltantes do cenário "a".
    df_sim_b : pd.DataFrame
        Dataset com NaN simulado. Cenário "b".
        Cenário "b": NaN introduzidos em todas as colunas.
    mascara_b : pd.DataFrame
        Máscara marcando as coordenadas de dados faltantes do cenário "b".
    """
    
    np.random.seed(seed)
    
    from utils import separar_colunas 
    metadados, variaveis = separar_colunas(df)
    
    # Remove todas as linhas com NaN
    df_ref = df.dropna().copy()
    
    # Identifica colunas que originalmente tinham NaN
    cols_com_nan = df.columns[df.isna().any()].tolist()
    cols_com_nan = [col for col in cols_com_nan if col not in metadados]
    
    # Todas as colunas numéricas (exceto metadados)
    todas_cols = [col for col in df_ref.columns if col not in metadados]
    
    # Cria dataset simulado
    df_sim_a = df_ref.copy()
    df_sim_b = df_ref.copy()

    n_remover = int(len(df_sim_a) * pct_remover)
    
    # Estabelece os cenários
    colunas_alvo_a = cols_com_nan
    mascara_a = pd.DataFrame(False, index=df_ref.index, columns=df_ref.columns)

    colunas_alvo_b = todas_cols
    mascara_b = pd.DataFrame(False, index=df_ref.index, columns=df_ref.columns)
    
    # Remove valores aleatoriamente apenas das colunas que tinham NaN original (cenário A)
    for col in colunas_alvo_a:
        indices = np.random.choice(df_sim_a.index, n_remover, replace=False)
        df_sim_a.loc[indices, col] = np.nan
        mascara_a.loc[indices, col] = True

    # Remove valores aleatoriamente de TODAS as colunas (exceto metadados) (cenário B)
    for col in colunas_alvo_b:
        indices = np.random.choice(df_sim_b.index, n_remover, replace=False)
        df_sim_b.loc[indices, col] = np.nan
        mascara_b.loc[indices, col] = True
    
    print(f"Cenário A: {len(colunas_alvo_a)} colunas, {pct_remover*100}% removido")
    print(f"Cenário B: {len(colunas_alvo_b)} colunas, {pct_remover*100}% removido")

    return df_sim_a, mascara_a, df_sim_b, mascara_b

# %% [markdown]
# # Métricas de Avaliação

# %% [markdown]
# Fluxograma simplificado:
# 0) Carregar dados
# 
# 1) Usar a função simular_dados_faltantes: É preciso especificar o dataframe verdadeiro e a porcentagem de dados a serem removidos (default 5%, valor utilizado para imputação via média e mediana). (i) Ela remove linhas com NaN, (ii) introduz NaN artificiais, (iii) retorna um dataframe com NaN artificiais apenas nas colunas que originalmente possuíam NaN (cenário "a"), (iv) retorna a máscara com as coordenadas desses NaN, (v) retorna um dataframe com NaN artificiais em todas as colunas, e (vi) a máscara com as coordenadas desses NaN.
# 
# 2) Fazer a imputação no dataframe artificial gerado na etapa anterior. É preciso especificar qual DataFrame, e se deseja-se que o DataFrame pós imputação apresente apenas as colunas que foram manipuladas (return_reduced=True)
# 
# 3) Fazer avaliação de imputação, utilizando a função avaliar_imputacao: É preciso especificar: (i) o dataframe real, (ii) o dataframe imputado, (iii) a máscara do dataframe imputado, (iv) o nome do método de imputação, e (v) qual cenário está sendo tratado ("a" ou "b").

# %%
def avaliar_imputacao(df_verdadeiro, df_imputado, mascara,
                        metodo, cenario, pct_faltantes=None):
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
    cenario : str
        Nome do cenário sendo avaliado:
        - "a" = avalia apenas colunas que tinham NaN original
        - "b" = avalia TODAS as colunas numéricas
    pct_faltantes : float, optional
        Porcentagem de dados faltantes (ex: 0.05 para 5%).
        Padrão: None
    
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
        - cenario: cenário avaliado
        - pct_faltantes: percentual de dados faltantes
    """
    
    from utils import separar_colunas
    metadados, variaveis = separar_colunas(df_verdadeiro)
    
    # Preparar dados
    df_verd_prep, df_imput_prep, colunas = preparar_para_metricas(
        df_verdadeiro, df_imputado, metadados
    )
    
    # Alinhar máscara ao mesmo índice e colunas usadas na métrica
    mascara_prep = mascara.loc[df_verd_prep.index, colunas]

    # Cria uma lista que será convertida em DataFrame, com todas as métricas de todas as variáveis

    linhas = []

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
            "cenario": cenario,
            "pct_faltantes": pct_faltantes
        })
    
    # Converte a lista "linhas" para DataFrame
    return pd.DataFrame(linhas)

# %%
def avaliacao_completa(df, seed=42):
    """
    Orquestra a avaliação completa de todos os métodos de imputação.
    
    Executa um pipeline completo que simula dados faltantes, aplica os três
    métodos de imputação (média, mediana, KNN) em ambos cenários e calcula
    as métricas de desempenho para cada combinação.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame completo (sem NaN) a ser usado como referência.
    seed : int, default 42
        Seed para reprodutibilidade dos dados simulados.
    
    Returns
    -------
    tabela_completa : pd.DataFrame
        DataFrame consolidado contendo métricas de todas as combinações:
        - média e mediana: 5% de dados removidos (cenários a e b)
        - KNN: 15% de dados removidos (cenários a e b)
        
        Colunas incluem: variavel, rmse, mae, r2, n_celulas, metodo, cenario, pct_faltantes
    
    Notes
    -----
    - Método média/mediana usa 5% de remoção de dados
    - Método KNN usa 15% de remoção de dados
    - Cada cenário é processado separadamente
    - Resultados são concatenados em um único DataFrame
    """

    from imputacao import mean_imput, median_imput, knn_imput
    
    ref = df.dropna().copy()

    partes = []

    # Para média e mediana
    sim_a, mascara_a, sim_b, mascara_b = simular_dados_faltantes(
        df, pct_remover=0.05, seed=seed
    )

    cenarios = [("a", sim_a, mascara_a), ("b", sim_b, mascara_b)]
    metodos = [("media", mean_imput), ("mediana", median_imput)]

    for cenario, sim, mascara in cenarios:
        for nome, func in metodos:
            imputado = func(sim, return_reduced=True)
            partes.append(
                avaliar_imputacao(ref, imputado, mascara, metodo=nome, cenario=cenario, pct_faltantes=0.05)
            )

    # Para KNN (mesmo loop de antes, só que com ajustes para acomodar o KNN)
    sim_a, mascara_a, sim_b, mascara_b = simular_dados_faltantes(
        df, pct_remover=0.15, seed=seed
    )

    cenarios = [("a", sim_a, mascara_a), ("b", sim_b, mascara_b)]
    metodos = [("knn", knn_imput)]

    for cenario, sim, mascara in cenarios:
        for nome, func in metodos:
            imputado = func(sim, return_reduced=True)
            partes.append(
                avaliar_imputacao(ref, imputado, mascara, metodo=nome, cenario=cenario, pct_faltantes=0.15)
            )
    
    return pd.concat(partes, ignore_index=True)


