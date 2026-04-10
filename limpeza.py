import pandas as pd
from utils import salvar

def limpeza(df, metadados, variaveis, threshold_missing=0.3, salvar_arquivo=True):
    """
    Remove dados corrompidos e colunas com muitos dados faltantes.
    
    df: DataFrame de interesse.
    metadados: Lista de colunas metadados.
    variaveis: Lista de colunas variáveis.
    threshold_missing: Remover colunas com % de dados faltantes acima disso (default 30%).
    salvar_arquivo: Salvar resultado (default True).
    
    Retorna: DataFrame limpo
    """
    # Evita modificar o original
    df = df.copy()  
    
    # Converte colunas variáveis para numeric, coerce non-numeric to NaN
    for coluna in variaveis:
        df[coluna] = pd.to_numeric(df[coluna], errors='coerce')
    
    # Remove colunas com dados faltantes acima do threshold
    thresh_count = len(df) * (1 - threshold_missing)
    df = df.dropna(axis=1, thresh=thresh_count)
    
    if salvar_arquivo:
        salvar(df, "dados_limpos")
    
    return df