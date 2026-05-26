# %%
import pandas as pd
import missingno as msno
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from utils import *

# %%
# Função que auxilia no salvamento de figuras
def salvar_plot(nome):
    """
    Salva a figura atual com o nome informado.

    Parameters
    ----------
    nome : str
        Nome base do arquivo de saída.

    Returns
    -------
    None
    """
    salvar_visualizacao(plt.gcf(), nome)
    plt.close()

# %% [markdown]
# # Boxplots

# %%
# Formatação dos boxplots

## Guarda as cores para cada parte do boxplot num dicionário
cores_customizadas = {      
    'boxes': 'Blue',
    'medians': 'Red',
    'whiskers': 'Yellow',
    'caps': 'Green'
}

estilo_outliers = dict(             # Determina a formatação dos pontos outliers
    markerfacecolor = 'magenta',    # Determina a cor dos pontos      
    marker = 'o',                   # Determina o formato dos pontos (no caso, círculos perfeitos)
    alpha = 0.5                     # Determina a opacidade / transparência dos pontos
)

# %%
def boxplot_tudo(df, plot=False, salvar=True):
    """
    Gera boxplots para todas as variáveis numa única figura.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse.
    plot : bool, default False
        Determina se o código deve gerar visualizações
        ao ser carregado.
    salvar : bool, default True
        Determina se o código deve salvar as visualizações
        ao ser carregado.

    Returns
    -------
    None
    """    
    if 'Estatistica' in df.columns:
        df.boxplot(by='Estatistica',        # Agrupa os dados por Estatística
            layout=(5, 6),                  # Organiza os boxplots em linhas e colunas
            figsize=(20,20),                # Determina o tamanho da figura
            sharey = False,                 # Faz com que diferentes variáveis apresentem diferentes escalas (mais apropriadas para cada uma)
            rot = 45,                       # Rotaciona os rótulos para evitar sobreposição
            color = cores_customizadas,     # Aplica as cores escolhidas
            flierprops = estilo_outliers,
            notch = True,
            )   
        # Aplica a formatação dos outliers escolhida
        plt.suptitle('Distribuição das Variáveis por Estatística (Escalas Independentes)', fontsize=16, y=1.02)

        # Ajusta o espaçamento entre os gráficos para não ficarem grudados
        plt.tight_layout()

        # Mostra a figura
        if plot:
            plt.show()
        
        # Salva a figura
        if salvar:
            salvar_plot('boxplot_tudo')
    else:
        print('Esse dataframe não apresenta coluna \'Estatistica\'. Talvez você tenha usado um dataframe pivotado?')

# %%
def boxplot_estatisticas(df, plot=False, salvar=True):
    """
    Cria boxplots separados para cada estatística (Min, Med, Max).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse.
    plot : bool, default False
        Determina se o código deve gerar visualizações
        ao ser carregado.
    salvar : bool, default True
        Determina se o código deve salvar as visualizações
        ao ser carregado.
    
    Returns
    -------
    None
    """    

    # Criando variaveis contendo cada tipo de estatistica (Min, Med, Max)

    cols_min = [coluna for coluna in df.columns if coluna.endswith('Min.')] 

    cols_med = [coluna for coluna in df.columns if coluna.endswith('Med.')] 

    cols_max = [coluna for coluna in df.columns if coluna.endswith('Max.')]

    if not cols_max and cols_med and cols_min:
        print("Não foram encontradas colunas finalizando com Min., Med. ou Max."
        "Talvez o DataFrames selecionado não esteja pivotado.")
        pass
    
    # Seleciona colunas com a estatística "Min."
    if not cols_min:
        print("Não foram encontradas colunas *_Min. no DataFrame")
        pass

    else:
        ax_min = df[cols_min].plot(kind='box',
                                    subplots=True,
                                    layout=(6,8),
                                    figsize=(20,20),
                                    sharey=False,
                                    color = cores_customizadas,
                                    flierprops = estilo_outliers
                                    )
        plt.suptitle('Distribuição dos valores mínimos de cada variável (Escalas independentes)', fontsize = 16, y = 1.02)

        plt.tight_layout()
        
        if plot:
            plt.show("boxplots_min")

        if salvar:
            salvar_plot("boxplots_min")

    # Seleciona colunas com a estatística "Med."
    if not cols_med:
        print("Não foram encontradas colunas *_Med. no DataFrame")
        pass
    else:
        ax_med = df[cols_med].plot(kind='box',
                                    subplots=True,
                                    layout=(6,8),
                                    figsize=(20,20),
                                    sharey=False,
                                    color = cores_customizadas,
                                    flierprops = estilo_outliers
                                    )
        plt.suptitle('Distribuição dos valores médios de cada variável (Escalas independentes)', fontsize = 16, y = 1.02)

        plt.tight_layout()

        if plot:
            plt.show("boxplots_med")

        if salvar:
            salvar_plot("boxplots_med")

    # Seleciona colunas com a estatística "Max."
    if not cols_max:
        print("Não foram encontradas colunas *_Max. no DataFrame")
        pass
    else:
        ax_max = df[cols_max].plot(kind='box',
                                    subplots=True,
                                    layout=(6,8),
                                    figsize=(20,20),
                                    sharey=False,
                                    color = cores_customizadas,
                                    flierprops = estilo_outliers
                                    )
        plt.suptitle('Distribuição dos valores máximos de cada variável (Escalas independentes)', fontsize = 16, y = 1.02)

        plt.tight_layout()

        if plot:
            plt.show("boxplots_max")

        if salvar:
            salvar_plot("boxplots_max")

# %% [markdown]
# # Dados Faltantes

# %%
# Seção biblioteca missingno
def visualizacoes_missingno(df, plot=False, salvar=True):
    """
    Gera visualizações dos dados faltantes no DataFrame
    de interesse com os métodos da biblioteca missingno.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse.
    plot : bool, default False
        Determina se o código deve gerar visualizações
        ao ser carregado.
    salvar : bool, default True
        Determina se o código deve salvar as visualizações
        ao ser carregado.

    Returns
    -------
    None
    """    
    ## Gera uma matriz representando dados faltantes/coluna
    msno.matrix(df)

    if salvar:
        salvar_plot("matrix_msno")

    if not plot:
        plt.close()

    ## Gera um gráfico de barras
    msno.bar(df) 

    if salvar:
        salvar_plot('bar_msno')

    if not plot:
        plt.close()

    ## Gera um mapa de calor
    msno.heatmap(df) 

    if salvar:
        salvar_plot('heatmap_msno')

    if not plot:
        plt.close()

    ## Gera um dendograma
    msno.dendrogram(df) 

    if salvar: 
        salvar_plot('dendogram_msno')

    if not plot:
        plt.close()   

# %%
# Taxa de faltantes / ano (0 → completo, 1 → tudo faltando)
def faltantes_ano(df, plot=False, salvar=True):
   """
   Gera um heatmap de dados faltantes ao longo dos anos.

   Parameters
   ----------
   df : pd.DataFrame
      DataFrame de interesse.
   plot : bool, default False
      Determina se o código deve gerar visualizações
      ao ser carregado.
   salvar : bool, default True
      Determina se o código deve salvar as visualizações
      ao ser carregado.

   Returns
   -------
   None
   """   
   # Separa colunas de data numa lista
   colunas_data = [coluna for coluna in df.columns if str(df[coluna].dtype).startswith('datetime')]

   missing_por_ano = (df.drop(columns=colunas_data).groupby(df["Data"].dt.year).apply(lambda x: x.isna().mean()*100))

   ## transpor para variáveis ficarem no eixo vertical
   missing_por_ano = missing_por_ano.T 

   plt.figure(figsize=(18, 10))
   sns.heatmap(
      missing_por_ano,
      cmap="RdBu_r",  # azul = completo | vermelho = faltante
      vmin=0,
      vmax=100,
      linewidths=0.5,
      linecolor="black"
   )

   plt.title("Taxa de dados faltantes (%) por variável e ano")
   plt.xlabel("Ano")
   plt.ylabel("Variável")

   if salvar:
      salvar_plot("faltantes_ano")

   if plot:
      plt.show()

# %%
# Taxa de faltantes / mês

def faltantes_mes(df, plot=False, salvar=True):
    """
    Gera um heatmap de dados faltantes em função dos meses.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse.
    plot : bool, default False
        Determina se o código deve gerar visualizações
        ao ser carregado.
    salvar : bool, default True
        Determina se o código deve salvar as visualizações
        ao ser carregado.

    Returns
    -------
    None
    """   
    # Separa colunas de data numa lista
    colunas_data = [coluna for coluna in df.columns if str(df[coluna].dtype).startswith('datetime')]

    missing_meses = (df.drop(columns=colunas_data)).groupby(df['Data'].dt.year).apply(lambda x: x.isna().sum())

    missing_meses = missing_meses.T

    plt.figure(figsize=(18, 10))
    sns.heatmap(
        missing_meses,
        cmap="RdBu_r",
        vmin=0,
        vmax=12,
        linewidths=0.5,
        linecolor="black",
        cbar_kws={"label": "Meses faltantes"}
    )

    plt.title("Meses faltantes por variável e ano")
    plt.xlabel("Ano")
    plt.ylabel("Variável")

    if salvar:
        salvar_plot('faltantes_mes')

    if plot:
        plt.show()

# %% [markdown]
# # Scatter Plots

# %%
def scatter_tudo(df, plot=False, salvar=True):
    """
    Gera scatterplots de todas as variáveis numa única figura.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse.
    plot : bool, default False
        Determina se o código deve gerar visualizações
        ao ser carregado.
    salvar : bool, default True
        Determina se o código deve salvar as visualizações
        ao ser carregado.
    
    Returns
    -------
    None
    """    

    fig, axes = plt.subplots(nrows=math.ceil(len(variaveis) / 4), 
                            ncols=4,
                            figsize=(20, 20))
    axes = axes.flatten()

    for i, col in enumerate(variaveis):
        sns.scatterplot(
            data=df,
            x='data_normalizada',
            y=col,
            hue='Estatistica',
            ax=axes[i]
        )
        axes[i].set_title(f'{col}')

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, labels, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 0.97),
                ncol=3,
                title='Estatística',
                fontsize=12,
                title_fontsize=14)
    
    for ax in axes:
        legenda = ax.get_legend()
        if legenda is not None:
            legenda.remove()

    for j in range(len(variaveis), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Scatterplots agrupados por Estatística (Escalas independentes)', 
                 fontsize=16,
                 y=1.0)
    
    plt.tight_layout()

    if salvar:
        salvar_visualizacao('scatterplots')

    if plot:
        plt.show()

# %% [markdown]
# # Correlação

# %%
def correlacao(df, method='pearson', plot=False, salvar=True):
    """
    Gera heatmaps de correlação.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse.
    method : str, default 'pearson'
        Método de cálculo de correlação. Opções:
        'pearson', 'kendall', 'spearman'.
    plot : bool, default False
        Determina se o código deve gerar visualizações
        ao ser carregado.
    salvar : bool, default True
        Determina se o código deve salvar as visualizações
        ao ser carregado.

    Returns
    -------
    None
    """    

    matriz = df[variaveis].corr(method=method)

    plt.figure(figsize=(20,20))

    sns.heatmap(data=matriz, 
                annot=True, 
                cmap='coolwarm', 
                fmt=".2f", 
                linewidths=0.5,
                vmin=-1,
                vmax=1
                )

    plt.title('Correlação de ' + str(method).title())
    plt.tight_layout()

    if salvar:
        salvar_visualizacao('correlacao_de', method)

    if plot:
        plt.show()

# %% [markdown]
# # Todas as Visualizações

# %%
def visualizacoes_todas(df, plot=False, salvar=True):
    """
    Gera todas as visualizações, exceto os boxplots de estatísticas:
    (biblioteca missingno, heatmap de dados faltantes por ano, 
    heatmap de dados, faltantes por mês, boxplots de todas as variáveis,
    scatterplots de todas as variáveis, heatmaps de correlação)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de interesse
    plot : bool, default False
        Determina se o código deve gerar visualizações
        ao ser carregado.
    salvar : bool, default True
        Determina se o código deve salvar as visualizações
        ao ser carregado.

    Returns
    -------
    None
    """        # 
    visualizacoes_missingno(df, plot=plot, salvar=salvar)
    faltantes_ano(df, plot=plot, salvar=salvar)
    faltantes_mes(df, plot=plot, salvar=salvar)
    boxplot_tudo(df, plot=plot, salvar=salvar)
    scatter_tudo(df, plot=plot, salvar=salvar)
    correlacao(df, plot=plot, salvar=salvar)


