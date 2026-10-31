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
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import math

from pcj.utils import salvar_visualizacao


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
    from pcj.utils import separar_colunas
    
    metadados, variaveis = separar_colunas(df)

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
        salvar_plot('scatterplots')

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
    from pcj.utils import separar_colunas
    metadados, variaveis = separar_colunas(df)

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
        salvar_plot('correlacao_de', method)

    if plot:
        plt.show()


# %% [markdown]
# # K-Means

# %%
def k_means_clusters(df, tabela, variaveis=None, eixos=None, nome="k_means_clusters", plot=False, salvar=True):
    """
    Gera a figura dos grupos formados pelo K-Means (função modelos_ML.k_means): cada
    mês é um ponto, com a cor e o marcador do seu grupo. Os grupos vêm prontos da
    tabela da k_means; a função só escolhe os eixos da figura (o "ângulo" da foto),
    então trocar os eixos não muda os grupos.

    Parameters
    ----------
    df : pd.DataFrame
        O mesmo DataFrame usado na k_means.
    tabela : pd.DataFrame
        Primeiro valor devolvido pela k_means: uma linha por mês, com a coluna "grupo"
        (<NA> nos meses não agrupados, que ficam fora da figura).
    variaveis : list, optional
        Lista com nomes das colunas identificadas como variaveis de df, by default None
    eixos : tuple of str, optional
        Duas variáveis de df para os eixos x e y (ex.: ("TURB.", "Cond.")): fácil de ler,
        mas mostra só 2 das variáveis. Se None, usa os dois primeiros componentes
        principais (PC1 e PC2), no mesmo espaço da k_means, com a fração da variação
        que cada um guarda no rótulo, by default None
    nome : str, default "k_means_clusters"
        Nome base do arquivo salvo.
    plot : bool, default False
        Determina se o código deve gerar visualizações
        ao ser carregado.
    salvar : bool, default True
        Determina se o código deve salvar as visualizações
        ao ser carregado.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Se a tabela tiver mais grupos do que a figura consegue distinguir (8).
    """
    if variaveis is None:
        from pcj.utils import separar_colunas
        _, variaveis = separar_colunas(df)

    # Cores e marcadores em ordem fixa (grupo 0, 1, 2...). O marcador diferente ajuda quem não distingue as cores
    cores = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    marcadores = ["o", "^", "s", "D", "v", "P", "X", "*"]
    cor_texto = "#0b0b0b"
    cor_eixos = "#898781"
    cor_grade = "#e1e0d9"

    # Só os meses agrupados (os outros têm grupo <NA>)
    grupos = tabela["grupo"].dropna()
    lista_grupos = sorted(grupos.unique())

    if len(lista_grupos) > len(cores):
        raise ValueError(f"A figura distingue até {len(cores)} grupos; a tabela tem {len(lista_grupos)}.")

    # 1. Coordenadas de cada mês nos eixos da figura
    if eixos is None:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        # O mesmo espaço da k_means: os meses agrupados, padronizados e reduzidos por PCA (95% da variância)
        X_nonan = df.loc[grupos.index, variaveis]

        scaler = StandardScaler()   # Coloca todas as variáveis na mesma escala, com média = 0 e desvio padrão = 1.
        X_scaled = scaler.fit_transform(X_nonan)

        pca = PCA(n_components=0.95, # Mantém 95% da variância
                random_state=0)
        X_pca = pca.fit_transform(X_scaled)

        # Os dois primeiros componentes são os que guardam mais variação: o melhor "ângulo" para uma figura plana
        x = X_pca[:, 0]
        y = X_pca[:, 1]

        variacao = pca.explained_variance_ratio_
        rotulo_x = f"PC1 ({variacao[0]:.0%} da variação)"
        rotulo_y = f"PC2 ({variacao[1]:.0%} da variação)"
    else:
        x = df.loc[grupos.index, eixos[0]].to_numpy()
        y = df.loc[grupos.index, eixos[1]].to_numpy()

        rotulo_x = eixos[0]
        rotulo_y = eixos[1]

    # 2. Figura: um conjunto de pontos por grupo
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, grupo in enumerate(lista_grupos):
        # True nos meses do grupo (to_numpy(dtype=bool): o grupo é "Int64", e a comparação devolve o booleano do pandas)
        no_grupo = (grupos == grupo).to_numpy(dtype=bool)

        ax.scatter(x[no_grupo], y[no_grupo], s=45, color=cores[i], marker=marcadores[i],
                   edgecolors="white", linewidths=0.6, label=f"grupo {grupo} ({no_grupo.sum()} meses)")

    # Títulos, eixos e grade
    ax.set_title(f"Grupos do K-Means (k = {len(lista_grupos)})", color=cor_texto)
    ax.set_xlabel(rotulo_x, color=cor_eixos)
    ax.set_ylabel(rotulo_y, color=cor_eixos)
    ax.legend(frameon=False)
    ax.tick_params(colors=cor_eixos)
    ax.grid(color=cor_grade, linewidth=0.8)
    ax.set_axisbelow(True)

    # Sem as bordas de cima e da direita
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Salva sem fechar a figura (o salvar_plot fecha), para ela ainda poder ser exibida
    if salvar:
        salvar_visualizacao(fig, nome)

    if plot:
        plt.show()
    else:
        plt.close(fig)


# %% [markdown]
# # K-Means - Escolha do número de grupos (k)

# %%
def curvas_selecao_k(resumo, nome="selecao_k", plot=False, salvar=True):
    """
    Gera a figura dos três critérios de escolha de k (saída da função
    modelos_ML.selecao_k): um painel para a inércia (método do cotovelo), um para a
    silhueta e um para o Calinski-Harabasz. Cada ponto é a média entre as seeds, e a
    barra de erro é o desvio padrão. O k apontado por cada critério é marcado com um
    círculo vazado e um texto.

    Parameters
    ----------
    resumo : pd.DataFrame
        Segundo valor devolvido por selecao_k: uma linha por k, com as colunas
        "<critério>_media", "<critério>_dp" e "indicado_<critério>".
    nome : str, default "selecao_k"
        Nome base do arquivo salvo (ex.: "selecao_k_tratada", para separar as bases).
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
    # Cores: série em azul; marcação do k e textos em tons neutros
    cor_serie = "#2a78d6"
    cor_texto = "#0b0b0b"
    cor_eixos = "#898781"
    cor_grade = "#e1e0d9"

    # Um painel por critério: (critério, título do painel, coluna que marca o k, rótulo da marcação)
    criterios = [
        ("inercia", "Inércia (método do cotovelo)", "indicado_cotovelo", "cotovelo"),
        ("silhueta", "Coeficiente de silhueta", "indicado_silhueta", "maior média"),
        ("calinski_harabasz", "Índice de Calinski-Harabasz", "indicado_calinski_harabasz", "maior média"),
    ]

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

    for i, (criterio, titulo, indicado, rotulo) in enumerate(criterios):
        ax = axs[i]

        # Média de cada k, com o desvio padrão entre as seeds como barra de erro
        ax.errorbar(
            resumo["k"],
            resumo[f"{criterio}_media"],
            yerr=resumo[f"{criterio}_dp"],
            color=cor_serie,
            linewidth=2,
            marker="o",
            markersize=6,
            capsize=3,
        )

        # k apontado pelo critério (a linha do resumo com True na coluna "indicado_...")
        linha = resumo[resumo[indicado]]

        if len(linha) == 0:
            # Só acontece no cotovelo: o kneed não achou um ponto de inflexão
            texto = "sem cotovelo"
        else:
            k_indicado = linha["k"].iloc[0]
            valor = linha[f"{criterio}_media"].iloc[0]
            texto = f"{rotulo}: k = {k_indicado}"

            # Círculo vazado em volta do ponto do k indicado
            ax.scatter(k_indicado, valor, s=250, facecolors="none", edgecolors=cor_texto, linewidths=1.5, zorder=3)

        # Texto no canto de cima, à direita (transform=ax.transAxes: posição em fração do painel, de 0 a 1),
        # com fundo branco para não se misturar às barras de erro
        ax.text(0.97, 0.95, texto, transform=ax.transAxes, ha="right", va="top", color=cor_texto,
                bbox={"facecolor": "white", "edgecolor": cor_grade, "boxstyle": "round,pad=0.3"})

        # Títulos, eixos e grade
        ax.set_title(titulo, color=cor_texto)
        ax.set_xlabel("Número de grupos (k)", color=cor_eixos)
        ax.set_ylabel("Média ± DP entre as seeds", color=cor_eixos)
        ax.set_xticks(resumo["k"])
        ax.margins(y=0.12)  # folga em cima e embaixo, para o círculo e o texto não encostarem na borda
        ax.tick_params(colors=cor_eixos)
        ax.grid(color=cor_grade, linewidth=0.8)
        ax.set_axisbelow(True)

        # Sem as bordas de cima e da direita
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Salva sem fechar a figura (o salvar_plot fecha), para ela ainda poder ser exibida
    if salvar:
        salvar_visualizacao(fig, nome)

    if plot:
        plt.show()
    else:
        plt.close(fig)


# %% [markdown]
# # Validação das imputações

# %%
def validacao_imputacoes(resumo, metrica="rmse", nome=None, plot=False, salvar=True):
    """
    Gera a figura de validação das imputações do experimento A: um painel por
    variável, na unidade dela, com uma barra por método de imputação. A altura da
    barra é a métrica calculada juntando as células de todas as rodadas
    (metricas_avaliacao.metricas_juntando), e a barra de erro é o desvio padrão
    da métrica entre as rodadas.

    Parameters
    ----------
    resumo : pd.DataFrame
        Uma linha por método × variável, com as colunas "metodo", "pct_faltantes",
        "variavel", "<metrica>" e "<metrica>_dp" (o resumo do notebook
        aplicacoes/imputacoes_outliers_tratados, gravado na aba "resumo" do
        imputacoes_outliers_tratados.xlsx).
    metrica : str, default "rmse"
        "rmse" ou "mae". O R² fica de fora porque não tem DP entre as rodadas.
    nome : str, optional
        Nome base do arquivo salvo. Se None, usa "validacao_imputacoes_<metrica>", by default None
    plot : bool, default False
        Determina se o código deve gerar visualizações
        ao ser carregado.
    salvar : bool, default True
        Determina se o código deve salvar as visualizações
        ao ser carregado.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Se a métrica não for "rmse" nem "mae".
    """
    if metrica not in ["rmse", "mae"]:
        raise ValueError(f'A métrica deve ser "rmse" ou "mae" (o R² não tem DP entre as rodadas); veio "{metrica}".')

    if nome is None:
        nome = f"validacao_imputacoes_{metrica}"

    # Cores por método (as mesmas da curvas_selecao_k e da k_means_clusters); textos e eixos em tons neutros
    cores = {"media": "#2a78d6", "mediana": "#eb6834", "knn": "#1baf7a"}
    nomes_metodos = {"media": "Média", "mediana": "Mediana", "knn": "KNN"}
    cor_texto = "#0b0b0b"
    cor_eixos = "#898781"
    cor_grade = "#e1e0d9"
    cor_erro = "#52514e"

    # 1. Métodos e variáveis na ordem em que aparecem no resumo (a ordem da base)
    metodos = list(resumo["metodo"].unique())
    variaveis = list(resumo["variavel"].unique())

    # 2. Grade de painéis: 4 colunas e quantas linhas forem precisas
    #    (squeeze=False: axs é sempre uma tabela de painéis, mesmo com uma linha só)
    n_colunas = 4
    n_linhas = math.ceil(len(variaveis) / n_colunas)
    fig, axs = plt.subplots(n_linhas, n_colunas, figsize=(12, 2.6 * n_linhas), squeeze=False)

    for n, variavel in enumerate(variaveis):
        # Linha e coluna do painel desta variável (// é a divisão inteira; % é o resto)
        ax = axs[n // n_colunas, n % n_colunas]

        # Uma barra por método, com o DP entre as rodadas como barra de erro
        for j, metodo in enumerate(metodos):
            linha = resumo[(resumo["metodo"] == metodo) & (resumo["variavel"] == variavel)].iloc[0]

            ax.bar(
                j,
                linha[metrica],
                yerr=linha[f"{metrica}_dp"],
                width=0.7,
                color=cores[metodo],
                capsize=3,
                error_kw={"ecolor": cor_erro, "elinewidth": 1},
            )

        # Título, eixos e grade. Os métodos não aparecem no eixo x: a cor diz qual é (ver a legenda)
        ax.set_title(variavel, color=cor_texto)
        ax.set_xticks([])
        ax.tick_params(colors=cor_eixos)
        ax.grid(axis="y", color=cor_grade, linewidth=0.8)
        ax.set_axisbelow(True)

        # Sem as bordas de cima e da direita
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 3. Painéis que sobram na grade (quando o número de variáveis não é múltiplo de 4) ficam em branco
    for n in range(len(variaveis), n_linhas * n_colunas):
        axs[n // n_colunas, n % n_colunas].axis("off")

    # 4. Uma legenda para a figura toda, com a taxa de falhas simulada de cada método (lida do resumo).
    #    Os quadrados (Rectangle) só servem para desenhar a cor de cada método na legenda
    quadrados = []
    rotulos = []
    for metodo in metodos:
        pct = resumo.loc[resumo["metodo"] == metodo, "pct_faltantes"].iloc[0]
        quadrados.append(plt.Rectangle((0, 0), 1, 1, color=cores[metodo]))
        rotulos.append(f"{nomes_metodos[metodo]} ({pct:.0%} de falhas)")

    fig.suptitle(f"{metrica.upper()} das imputações ± DP entre as rodadas (na unidade de cada variável)",
                 color=cor_texto, y=0.995)
    fig.legend(quadrados, rotulos, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(metodos), frameon=False)

    # rect deixa livre a faixa de cima da figura, onde ficam o título e a legenda
    plt.tight_layout(rect=(0, 0, 1, 0.93))

    # Salva sem fechar a figura (o salvar_plot fecha), para ela ainda poder ser exibida
    if salvar:
        salvar_visualizacao(fig, nome)

    if plot:
        plt.show()
    else:
        plt.close(fig)


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
    """
    visualizacoes_missingno(df, plot=plot, salvar=salvar)
    faltantes_ano(df, plot=plot, salvar=salvar)
    faltantes_mes(df, plot=plot, salvar=salvar)
    boxplot_tudo(df, plot=plot, salvar=salvar)
    scatter_tudo(df, plot=plot, salvar=salvar)
    correlacao(df, plot=plot, salvar=salvar)
