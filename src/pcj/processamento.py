# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv (3.14.2)
#     language: python
#     name: python3
# ---

# %%
import os
from datetime import datetime
import pandas as pd

from pcj.utils import separar_colunas, salvar

from pcj.format import casas_decimais


# %% [markdown]
# # Prep Dados Brutos

# %%
def prep_brutos(df, salvar_arquivo=True):
    """
    Extrai data normalizada (sem horário), reseta index, e renomeia.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com dados brutos.
    salvar_arquivo : bool, default True
        Determina se um arquivo Excel com os dados brutos deve ser salvo.

    Returns
    -------
    df : pd.DataFrame
        Versão alterada do DataFrame fornecido.
    metadados : list
        Lista com os nomes das colunas identificadas como metadados de df.
    variaveis : list
        Lista com nomes das colunas identificadas como variáveis de df.
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

    if salvar_arquivo:
        salvar(df, "dados_prontos")

    return df, metadados, variaveis


# %% [markdown]
# # Limpeza

# %%
def limpeza(df, metadados=None, variaveis=None, threshold_missing=0.3, salvar_arquivo=True):
    """
    Remove dados corrompidos e colunas com muitos dados faltantes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser limpo.
    metadados : list, default None
        Lista com os nomes das colunas identificadas como metadados de df.
    variaveis : list, default None
        Lista com nomes das colunas identificadas como variáveis de df.
    threshold_missing : float, default 0.3
        Determina a porcentagem máxima tolerável de valores faltantes numa coluna.
    salvar_arquivo : bool, default True
        Determina se um arquivo Excel com o DataFrame limpo deve ser gerado.

    Returns
    -------
    df : pd.DataFrame
        Versão alterada do DataFrame fornecido.
    metadados : list
        Lista com os nomes das colunas identificadas como metadados de df.
    variaveis : list
        Lista com nomes das colunas identificadas como variáveis de df.
    """    
    # Evita modificar o original
    df = df.copy()

    # Determina as listas de metadados e variaveis com base em df
    if metadados is None or variaveis is None:
        from pcj.utils import separar_colunas
        metadados, variaveis = separar_colunas(df)  
    
    # Converte colunas variáveis para numeric, coerce non-numeric to NaN
    for coluna in variaveis:
        df[coluna] = pd.to_numeric(df[coluna], errors='coerce')
    
    # Remove colunas com dados faltantes acima do threshold
    thresh_count = len(df) * (1 - threshold_missing)
    df = df.dropna(axis=1, thresh=thresh_count)

    # Salva as novas listas de metadados e variaveis após a limpeza
    metadados, variaveis = separar_colunas(df)
    
    if salvar_arquivo:
        salvar(df, "dados_limpos")
    
    return df, metadados, variaveis


# %% [markdown]
# # Decomposição Estatísticas
#
# Quero decompor os dados em 3: Uma seção "Mínimo", outra seção "Média" e outra "Máximo"

# %% [markdown]
# eu provavelmente vou precisar escrever uma lógica daquele naipe que aplica linha a linha para criar o os dataframes derivados

# %%
def decompor_estatisticas(df, estatistica=False):
    """
    Decompõe a base de dados em 3, uma para cada estatística:
    1 base de dados para os valores mínimos de cada mês, 1 para os
    médios, e 1 para os máximos

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com todos os valores de todas estatísticas.

    Returns
    -------
    df_min : pd.DataFrame
        DataFrame com valores mínimos
    df_med : pd.DataFrame
        DataFrame com valores médios
    df_max : pd.DataFrame
        DataFrame com valores máximos
    """
    from pcj.utils import dados

    df_completo = df.copy()

    df_min, df_med, df_max = df_completo[df_completo['Estatistica'] == 'Min.'], df_completo[df_completo['Estatistica'] == 'Med.'], df_completo[df_completo['Estatistica'] == 'Max.']

    if estatistica == "min":
        return df_min

    elif estatistica == "med":
        return df_med

    elif estatistica == "max":
        return df_max

    elif estatistica == "min,med":
        return df_min, df_med

    elif estatistica == "min,max":
        return df_min, df_max

    elif estatistica == "med,max":
        return df_med, df_max
    
    else:
        return df_min, df_med, df_max
