import pandas as pd
from sklearn.impute import KNNImputer
from utils import salvar

def filtro_colunas_5(df, variaveis, max_missing=0.05):
    """
    Seleciona colunas com 5% ou menos de dados faltantes
    """
    fracao_faltantes = df[variaveis].isna().mean()
    return fracao_faltantes[fracao_faltantes <= max_missing].index.tolist()

def mean_imput(df, variaveis, max_missing=0.05, salvar_arquivo=True):
    df = df.copy()
    cols = filtro_colunas_5(df, variaveis, max_missing)
    df[cols] = df[cols].fillna(df[cols].mean())
    if salvar_arquivo:
        salvar(df, "imput_mean")
    return df

def median_imput(df, variaveis, max_missing=0.05, salvar_arquivo=True):
    df = df.copy()
    cols = filtro_colunas_5(df, variaveis, max_missing)
    df[cols] = df[cols].fillna(df[cols].median())
    if salvar_arquivo:
        salvar(df, "imput_median")
    return df

def knn_imput(df, variaveis, salvar_arquivo=True):
    """
    Imputação via KNN (método do sklearn)
    """
    df = df.copy()

    imputer = KNNImputer(n_neighbors=2, weights="uniform")

    # Garante valores numéricos
    X = df[variaveis].astype(float)

    df[variaveis] = imputer.fit_transform(X)

    # Verifica se ainda há valores faltantes nas colunas imputadas
    print("Valores faltantes após imputação KNN:")
    print(df[variaveis].isna().sum())

    if salvar_arquivo:
        salvar(df, "imput_knn")

    return df