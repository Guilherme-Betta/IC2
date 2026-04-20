import pandas as pd
from utils import *
from analise_descritiva import casas_decimais

def prep_brutos(df):
    """
    Prepara dados brutos: extrai data normalizada, reseta index, e renomeia.
    df: DataFrame de dados brutos a serem preparado (deve ter coluna 'Data', com dtype datetime)
    """

    # Trabalha em uma cópia para evitar modificar o original
    df = df.copy()
    
    # Cria coluna de data normalizada
    df['data_normalizada'] = df['Data'].dt.date
    
    # Move a coluna de data normalizada para a posição 1
    col = df.pop('data_normalizada')
    df.insert(1, 'data_normalizada', col)
    
    # Reseta index e renomeia a coluna da Index gerada automaticamente pelo pandas
    df = df.reset_index()
    df.rename(columns={'index': 'Index'}, inplace=True)

    metadados, variaveis = separar_colunas(df)
    
    # Armazena o número de casas decimais utilizadas por variável na base de dados. 
    # É útil implementar esse método enquanto a base ainda está alterada, portanto,
    # Ela está sendo colocada nessa função dedicada aos dados brutos.

    n_casas_decimais = {}

    for i in variaveis:
        n_casas_decimais[i] = casas_decimais(df[i])

    return df, metadados, variaveis, n_casas_decimais

def limpeza(df, metadados, variaveis, threshold_missing=0.3, salvar_arquivo=True):
    """
    Remove dados corrompidos e colunas com muitos dados faltantes.
    
    df: DataFrame de interesse;
    metadados: Lista de colunas metadados;
    variaveis: Lista de colunas variáveis;
    threshold_missing: Remover colunas com % de dados faltantes acima disso (default 30%);
    salvar_arquivo: Salvar resultado (default True).
    
    Retorna: 
    df = DataFrame limpo; 
    metadados = Novo conjunto de metadados;
    variaveis = Novo conjunto de variáveis. 
    """
    # Evita modificar o original
    df = df.copy()  
    
    # Converte colunas variáveis para numeric, coerce non-numeric to NaN
    for coluna in variaveis:
        df[coluna] = pd.to_numeric(df[coluna], errors='coerce')
    
    # Remove colunas com dados faltantes acima do threshold
    thresh_count = len(df) * (1 - threshold_missing)
    df = df.dropna(axis=1, thresh=thresh_count)

    metadados, variaveis = separar_colunas(df)
    
    if salvar_arquivo:
        salvar(df, "dados_limpos")
    
    return df, metadados, variaveis