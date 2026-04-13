import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from utils import salvar_visualizacao

# Formatação dos boxplots
cores_customizadas = {      
    'boxes': 'Blue',
    'medians': 'Red',
    'whiskers': 'Yellow',
    'caps': 'Green'
}

estilo_outliers = dict(             
    markerfacecolor='magenta',    
    marker='o',                   
    alpha=0.5                     
)

# Função que auxilia no salvamento de figuras
def salvar_plot(nome):
    salvar_visualizacao(plt.gcf(), nome)
    plt.show()
    plt.close()

def boxplot_tudo(df):
    """
    Gera boxplots para todas as variáveis numa única figura.
    df: Seleciona o dataframe em que se deseja.
    """
    if 'Estatistica' in df.columns:
        df.boxplot(by='Estatistica',             
            layout=(5, 6),                  
            figsize=(20,20),                
            sharey=False,                 
            rot=45,                       
            color=cores_customizadas,     
            flierprops=estilo_outliers,
            notch=True,
            )   
        plt.suptitle('Distribuição das Variáveis por Estatística (Escalas Independentes)', fontsize=16, y=1.02)
        plt.tight_layout()
        salvar_plot('boxplot_tudo')
    else:
        print('Esse dataframe não apresenta coluna \'Estatistica\'. Talvez você tenha usado um dataframe pivotado?')

def boxplot_estatisticas(df):
    """
    Cria boxplots separados para cada estatística (Min, Med, Max)
    """
    cols_min = [coluna for coluna in df.columns if coluna.endswith('Min.')] 
    cols_med = [coluna for coluna in df.columns if coluna.endswith('Med.')] 
    cols_max = [coluna for coluna in df.columns if coluna.endswith('Max.')]

    # Min
    df[cols_min].plot(kind='box', subplots=True, layout=(6,8), figsize=(20,20), sharey=False, color=cores_customizadas, flierprops=estilo_outliers)
    plt.suptitle('Distribuição dos valores mínimos de cada variável (Escalas independentes)', fontsize=16, y=1.02)
    plt.tight_layout()
    salvar_plot("boxplots_min")

    # Med
    df[cols_med].plot(kind='box', subplots=True, layout=(6,8), figsize=(20,20), sharey=False, color=cores_customizadas, flierprops=estilo_outliers)
    plt.suptitle('Distribuição dos valores médios de cada variável (Escalas independentes)', fontsize=16, y=1.02)
    plt.tight_layout()
    salvar_plot("boxplots_med")

    # Max
    df[cols_max].plot(kind='box', subplots=True, layout=(6,8), figsize=(20,20), sharey=False, color=cores_customizadas, flierprops=estilo_outliers)
    plt.suptitle('Distribuição dos valores máximos de cada variável (Escalas independentes)', fontsize=16, y=1.02)
    plt.tight_layout()
    salvar_plot("boxplots_max")

def visualizacoes_missingno(df):
    """
    Gera visualizações dos dados faltantes no DataFrame de interesse com os métodos da biblioteca missingno.
    df: Seleciona o DataFrame de interesse.
    """
    msno.matrix(df) 
    salvar_plot("matrix_msno")

    msno.bar(df) 
    salvar_plot('bar_msno')

    msno.heatmap(df) 
    salvar_plot('heatmap_msno')

    msno.dendrogram(df)  
    salvar_plot('dendogram_msno')

def faltantes_ano(df):
    """
    Gera um heatmap de dados faltantes ao longo dos anos.
    df: DataFrame de interesse.
    """
    colunas_data = [coluna for coluna in df.columns if str(df[coluna].dtype).startswith('datetime')]
    missing_por_ano = (df.drop(columns=colunas_data).groupby(df["Data"].dt.year).apply(lambda x: x.isna().mean()*100))
    missing_por_ano = missing_por_ano.T 

    plt.figure(figsize=(18, 10))
    sns.heatmap(missing_por_ano, cmap="RdBu_r", vmin=0, vmax=100, linewidths=0.5, linecolor="black")
    plt.title("Taxa de dados faltantes (%) por variável e ano")
    plt.xlabel("Ano")
    plt.ylabel("Variável")
    salvar_plot("faltantes_ano")

def faltantes_mes(df):
    """
    Gera um heatmap de dados faltantes em função dos meses.
    df: DataFrame de interesse.
    """
    colunas_data = [coluna for coluna in df.columns if str(df[coluna].dtype).startswith('datetime')]
    missing_meses = (df.drop(columns=colunas_data)).groupby(df['Data'].dt.year).apply(lambda x: x.isna().sum())
    missing_meses = missing_meses.T

    plt.figure(figsize=(18, 10))
    sns.heatmap(missing_meses, cmap="RdBu_r", vmin=0, vmax=12, linewidths=0.5, linecolor="black", cbar_kws={"label": "Meses faltantes"})
    plt.title("Meses faltantes por variável e ano")
    plt.xlabel("Ano")
    plt.ylabel("Variável")
    salvar_plot('faltantes_mes')

def visualizacoes_todas(df):
    boxplot_tudo(df)
    boxplot_estatisticas(df)
    visualizacoes_missingno(df)
    faltantes_ano(df)
    faltantes_mes(df)