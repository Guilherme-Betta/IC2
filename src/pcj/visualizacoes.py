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
    Salva a figura atual com o nome informado, sem fechá-la: quem chama
    decide se a figura é mostrada (plt.show) ou fechada (plt.close).

    Parameters
    ----------
    nome : str
        Nome base do arquivo de saída.

    Returns
    -------
    None
    """
    salvar_visualizacao(plt.gcf(), nome)


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

        # Salva antes de mostrar (no notebook, o plt.show() fecha a figura)
        if salvar:
            salvar_plot('boxplot_tudo')

        # Mostra a figura ou a fecha
        if plot:
            plt.show()
        else:
            plt.close()
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

    if not cols_max and not cols_med and not cols_min:
        print("Não foram encontradas colunas finalizando com Min., Med. ou Max."
        " Talvez o DataFrame selecionado não esteja pivotado.")
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
        
        # Salva antes de mostrar (no notebook, o plt.show() fecha a figura)
        if salvar:
            salvar_plot("boxplots_min")

        # Mostra a figura ou a fecha
        if plot:
            plt.show()
        else:
            plt.close()

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

        # Salva antes de mostrar (no notebook, o plt.show() fecha a figura)
        if salvar:
            salvar_plot("boxplots_med")

        # Mostra a figura ou a fecha
        if plot:
            plt.show()
        else:
            plt.close()

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

        # Salva antes de mostrar (no notebook, o plt.show() fecha a figura)
        if salvar:
            salvar_plot("boxplots_max")

        # Mostra a figura ou a fecha
        if plot:
            plt.show()
        else:
            plt.close()


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

    if plot:
        plt.show()
    else:
        plt.close()

    ## Gera um gráfico de barras
    msno.bar(df) 

    if salvar:
        salvar_plot('bar_msno')

    if plot:
        plt.show()
    else:
        plt.close()

    ## Gera um mapa de calor
    msno.heatmap(df) 

    if salvar:
        salvar_plot('heatmap_msno')

    if plot:
        plt.show()
    else:
        plt.close()

    ## Gera um dendograma
    msno.dendrogram(df) 

    if salvar: 
        salvar_plot('dendogram_msno')

    if plot:
        plt.show()
    else:
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
   else:
      plt.close()


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
    else:
        plt.close()


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
    else:
        plt.close()


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
        salvar_plot('correlacao_de_' + str(method))

    if plot:
        plt.show()
    else:
        plt.close()


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

    # Salva sem fechar a figura, para ela ainda poder ser exibida
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

    # Salva sem fechar a figura, para ela ainda poder ser exibida
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

    # Salva sem fechar a figura, para ela ainda poder ser exibida
    if salvar:
        salvar_visualizacao(fig, nome)

    if plot:
        plt.show()
    else:
        plt.close(fig)


# %% [markdown]
# # Com × sem tratamento de outliers (imputação)

# %%
def rodadas_ganhas_barras(rodadas_ganhas, metricas=("rmse", "mae", "r2"), nome="rodadas_ganhas",
                          plot=False, salvar=True):
    """
    Gera a figura-resumo do com × sem tratamento de outliers na imputação: em quantas
    comparações (rodadas × variáveis) cada base foi melhor, por método e taxa de falhas.
    Um painel por métrica, com uma barra empilhada por método × taxa (base tratada melhor,
    empate e base não tratada melhor), somando as variáveis.

    Parameters
    ----------
    rodadas_ganhas : pd.DataFrame
        Uma linha por taxa × métrica × método × variável, com as colunas "pct_faltantes",
        "metrica", "metodo", "variavel", "tratada_melhor", "empate" e "nao_tratada_melhor"
        (a aba "rodadas_ganhas" do imputacoes_mesma_prova_nao_tratados.xlsx).
    metricas : tuple of str, default ("rmse", "mae", "r2")
        Métricas desenhadas, uma por painel, na ordem dada.
    nome : str, default "rodadas_ganhas"
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
    """
    # Cores: tratada melhor em azul, empate em cinza, não tratada melhor em vermelho;
    # textos e eixos em tons neutros
    cores = {"tratada_melhor": "#2a78d6", "empate": "#c3c2b7", "nao_tratada_melhor": "#e34948"}
    cores_numeros = {"tratada_melhor": "white", "empate": "#0b0b0b", "nao_tratada_melhor": "white"}
    nomes_metodos = {"media": "Média", "mediana": "Mediana", "knn": "KNN"}
    nomes_metricas = {"rmse": "RMSE", "mae": "MAE", "r2": "R²"}
    partes = ["tratada_melhor", "empate", "nao_tratada_melhor"]
    cor_texto = "#0b0b0b"
    cor_secundaria = "#52514e"
    cor_eixos = "#898781"

    # 1. Métodos e taxas na ordem em que aparecem na tabela
    metodos = list(rodadas_ganhas["metodo"].unique())
    taxas = list(rodadas_ganhas["pct_faltantes"].unique())

    # 2. Uma barra por método × taxa, de cima para baixo: a posição y diminui a cada barra,
    #    e um espaço extra separa os métodos
    barras = []
    posicoes = []
    rotulos_taxas = []
    y = 0
    for metodo in metodos:
        for taxa in taxas:
            barras.append((metodo, taxa))
            posicoes.append(y)
            rotulos_taxas.append(f"{taxa:.0%}")
            y = y - 1
        y = y - 0.9

    fig, axs = plt.subplots(1, len(metricas), figsize=(5 * len(metricas), 6.2), sharey=True, squeeze=False)

    for i, metrica in enumerate(metricas):
        ax = axs[0, i]

        # 3. Soma das variáveis: uma linha por método e taxa (o índice é o par (método, taxa))
        da_metrica = rodadas_ganhas[rodadas_ganhas["metrica"] == metrica]
        soma = da_metrica.groupby(["metodo", "pct_faltantes"])[partes].sum()

        # Comparações em cada barra (rodadas × variáveis): a soma das três partes, a mesma em todas
        total = soma.iloc[0].sum()

        # 4. Barras empilhadas: cada parte começa onde a anterior terminou (left)
        inicio = [0] * len(barras)
        for parte in partes:
            for b, (metodo, taxa) in enumerate(barras):
                valor = soma.loc[(metodo, taxa), parte]
                ax.barh(posicoes[b], valor, left=inicio[b], height=0.75, color=cores[parte],
                        edgecolor="white", linewidth=1.5)

                # Número dentro da parte, só se ela tiver espaço (8% da barra ou mais).
                # O fundo (bbox) da cor da parte esconde a linha da metade atrás do número
                if valor >= 0.08 * total:
                    ax.text(inicio[b] + valor / 2, posicoes[b], str(valor), ha="center", va="center",
                            fontsize=9, color=cores_numeros[parte], zorder=4,
                            bbox={"facecolor": cores[parte], "edgecolor": "none", "pad": 1})

                inicio[b] = inicio[b] + valor

        # 5. Linha da metade: à direita dela, a não tratada foi melhor na maioria das comparações
        ax.axvline(total / 2, color=cor_texto, linewidth=1, linestyle="--", zorder=3)
        ax.text(total / 2, posicoes[0] + 0.6, f"metade ({total // 2})", ha="center", va="bottom",
                fontsize=8, color=cor_secundaria, zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})

        # Título, eixos e rótulos das taxas
        ax.set_title(nomes_metricas[metrica], color=cor_texto, pad=16)
        ax.set_xlim(0, total)
        ax.set_xticks([0, total // 4, total // 2, 3 * total // 4, total])
        ax.set_xlabel(f"comparações (rodadas × variáveis, de {total})", color=cor_eixos)
        ax.set_yticks(posicoes)
        ax.set_yticklabels(rotulos_taxas)
        ax.tick_params(colors=cor_eixos, length=0)
        ax.tick_params(axis="y", colors=cor_texto)

        # Sem as bordas de cima, da direita e da esquerda
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    # 6. Nome de cada método à esquerda do grupo de barras dele, no primeiro painel.
    #    get_yaxis_transform: x em fração do painel (negativo = fora, à esquerda) e y nas posições das barras
    for m, metodo in enumerate(metodos):
        meio = sum(posicoes[m * len(taxas): (m + 1) * len(taxas)]) / len(taxas)
        axs[0, 0].text(-0.16, meio, nomes_metodos[metodo], transform=axs[0, 0].get_yaxis_transform(),
                       ha="right", va="center", color=cor_texto, fontweight="bold")

    # 7. Uma legenda para a figura toda. Os quadrados (Rectangle) só desenham as cores
    quadrados = []
    for parte in partes:
        quadrados.append(plt.Rectangle((0, 0), 1, 1, color=cores[parte]))
    rotulos = ["tratada melhor", "empate", "não tratada melhor"]
    fig.suptitle("Com × sem tratamento de outliers: em quantas comparações cada base foi melhor",
                 color=cor_texto, y=0.995)
    fig.legend(quadrados, rotulos, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False)
    fig.text(0.5, 0.895, "Melhor: erro menor no RMSE e no MAE; R² maior no R²", ha="center",
             color=cor_secundaria, fontsize=9)

    # rect deixa livre a faixa da esquerda (nomes dos métodos) e a de cima (título e legenda)
    plt.tight_layout(rect=(0.06, 0, 1, 0.88))

    # Salva sem fechar a figura, para ela ainda poder ser exibida
    if salvar:
        salvar_visualizacao(fig, nome)

    if plot:
        plt.show()
    else:
        plt.close(fig)


# %%
def diferenca_mapa_calor(comparacao, metricas=("rmse", "mae"), limite=10, nome="diferenca_mapa_calor",
                         plot=False, salvar=True):
    """
    Gera o mapa de calor do com × sem tratamento de outliers na imputação: a diferença
    do erro entre as bases (tratada − não tratada), em % do erro da não tratada, para cada
    variável (linhas) e cada método × taxa de falhas (colunas). Um painel por métrica.
    Azul: a tratada errou menos; vermelho: a não tratada errou menos.

    Parameters
    ----------
    comparacao : pd.DataFrame
        Uma linha por taxa × métrica × método × variável, com as colunas "pct_faltantes",
        "metrica", "metodo", "variavel", "nao_tratada" e "diferenca" (a aba "comparacao"
        do imputacoes_mesma_prova_nao_tratados.xlsx).
    metricas : tuple of str, default ("rmse", "mae")
        Métricas desenhadas, uma por painel. Só métricas de erro: no R², perto de 0, a
        diferença em % não tem sentido.
    limite : float, default 10
        Diferença (em %) que recebe a cor mais forte; acima disso, a cor não muda, mas o
        número escrito na célula é o valor real.
    nome : str, default "diferenca_mapa_calor"
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
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Cores: azul (a tratada erra menos), cinza claro no zero e vermelho (a não tratada erra menos)
    cmap = LinearSegmentedColormap.from_list("diferenca", ["#2a78d6", "#f0efec", "#e34948"])
    nomes_metodos = {"media": "Média", "mediana": "Mediana", "knn": "KNN"}
    nomes_metricas = {"rmse": "RMSE", "mae": "MAE"}
    cor_texto = "#0b0b0b"
    cor_eixos = "#898781"

    # 1. Métodos, taxas e variáveis na ordem em que aparecem na tabela
    metodos = list(comparacao["metodo"].unique())
    taxas = list(comparacao["pct_faltantes"].unique())
    variaveis = list(comparacao["variavel"].unique())

    fig, axs = plt.subplots(1, len(metricas), figsize=(6.5 * len(metricas), 8.6), squeeze=False)

    for i, metrica in enumerate(metricas):
        ax = axs[0, i]
        da_metrica = comparacao[comparacao["metrica"] == metrica]

        # 2. Tabela do mapa: uma linha por variável, uma coluna por método × taxa,
        #    com a diferença em % do erro da não tratada
        mapa = []
        for variavel in variaveis:
            linha_mapa = []
            for metodo in metodos:
                for taxa in taxas:
                    linha = da_metrica[(da_metrica["variavel"] == variavel) & (da_metrica["metodo"] == metodo)
                                       & (da_metrica["pct_faltantes"] == taxa)].iloc[0]
                    linha_mapa.append(100 * linha["diferenca"] / linha["nao_tratada"])
            mapa.append(linha_mapa)

        # 3. Mapa de cores: vmin e vmax fixam a escala em -limite a +limite (o 0 fica no meio, em cinza)
        imagem = ax.imshow(mapa, cmap=cmap, vmin=-limite, vmax=limite, aspect="auto")

        # 4. O número de cada célula, com vírgula. Em células de cor forte, o número fica branco
        for n_linha in range(len(variaveis)):
            for n_coluna in range(len(metodos) * len(taxas)):
                valor = mapa[n_linha][n_coluna]
                if abs(valor) < 0.05:
                    valor = 0.0    # evita "-0,0"
                cor_numero = "white" if abs(valor) > 0.6 * limite else cor_texto
                ax.text(n_coluna, n_linha, f"{valor:.1f}".replace(".", ","), ha="center", va="center",
                        fontsize=7.5, color=cor_numero)

        # 5. Linhas brancas entre os métodos (cada método ocupa len(taxas) colunas)
        for m in range(1, len(metodos)):
            ax.axvline(m * len(taxas) - 0.5, color="white", linewidth=4)

        # 6. Taxas em cima de cada coluna; nome do método e da métrica acima delas
        rotulos_taxas = []
        for metodo in metodos:
            for taxa in taxas:
                rotulos_taxas.append(f"{taxa:.0%}")
        ax.set_xticks(range(len(metodos) * len(taxas)))
        ax.set_xticklabels(rotulos_taxas, fontsize=8, color=cor_texto)
        ax.xaxis.tick_top()
        for m, metodo in enumerate(metodos):
            ax.text(m * len(taxas) + (len(taxas) - 1) / 2, -1.3, nomes_metodos[metodo], ha="center",
                    color=cor_texto)
        ax.text((len(metodos) * len(taxas) - 1) / 2, -2.4, nomes_metricas[metrica], ha="center",
                color=cor_texto, fontweight="bold")

        ax.set_yticks(range(len(variaveis)))
        ax.set_yticklabels(variaveis, color=cor_texto)
        ax.tick_params(length=0)

        # Sem bordas em volta do mapa
        for lado in ax.spines.values():
            lado.set_visible(False)

    fig.suptitle("Com × sem tratamento de outliers: diferença do erro, em %, por variável, método e taxa",
                 color=cor_texto, y=0.99)

    # Espaço entre os painéis (wspace) e, à direita, para a barra de cores
    fig.subplots_adjust(left=0.08, right=0.87, top=0.83, bottom=0.03, wspace=0.28)

    # 7. Barra de cores num eixo próprio, à direita dos mapas ([esquerda, baixo, largura, altura],
    #    em fração da figura), para não tomar o espaço deles
    eixo_barra = fig.add_axes([0.9, 0.3, 0.012, 0.4])
    barra = fig.colorbar(imagem, cax=eixo_barra)
    barra.set_label("tratada − não tratada (% do erro da não tratada)\nazul: a tratada errou menos",
                    color=cor_eixos, fontsize=8)
    barra.ax.tick_params(labelsize=8, colors=cor_eixos)
    barra.outline.set_visible(False)

    # Salva sem fechar a figura, para ela ainda poder ser exibida
    if salvar:
        salvar_visualizacao(fig, nome)

    if plot:
        plt.show()
    else:
        plt.close(fig)


# %%
def celulas_que_mudam_barras(celulas_que_mudam, nome="celulas_que_mudam", plot=False, salvar=True):
    """
    Gera a figura de quanto o valor imputado muda quando os meses com valores extremos
    entram na base (base não tratada × tratada, nas mesmas células): uma barra por
    método × taxa de falhas, com a mudança média em desvios padrão (DP) da variável, e
    a % de células em que o valor imputado é o mesmo nas duas bases.

    Parameters
    ----------
    celulas_que_mudam : pd.DataFrame
        Uma linha por método × taxa × variável, com as colunas "metodo", "pct_faltantes",
        "variavel", "n_celulas", "pct_iguais" e "mudanca_media_dp" (a aba
        "celulas_que_mudam" do imputacoes_mesma_prova_nao_tratados.xlsx).
    nome : str, default "celulas_que_mudam"
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
    """
    # Cores por método (as mesmas da validacao_imputacoes); textos e eixos em tons neutros
    cores = {"media": "#2a78d6", "mediana": "#eb6834", "knn": "#1baf7a"}
    nomes_metodos = {"media": "Média", "mediana": "Mediana", "knn": "KNN"}
    cor_texto = "#0b0b0b"
    cor_secundaria = "#52514e"
    cor_eixos = "#898781"
    cor_grade = "#e1e0d9"

    # 1. Métodos e taxas na ordem em que aparecem na tabela
    metodos = list(celulas_que_mudam["metodo"].unique())
    taxas = list(celulas_que_mudam["pct_faltantes"].unique())

    fig, ax = plt.subplots(figsize=(8, 5.4))

    # 2. Uma barra por método × taxa, de cima para baixo, com um espaço extra entre os métodos
    posicoes = []
    rotulos = []
    meios = []
    y = 0
    for metodo in metodos:
        posicoes_metodo = []
        for taxa in taxas:
            grupo = celulas_que_mudam[(celulas_que_mudam["metodo"] == metodo)
                                      & (celulas_que_mudam["pct_faltantes"] == taxa)]

            # Médias de todas as células do grupo: cada variável pesa pelo número de células dela
            n = grupo["n_celulas"].sum()
            mudanca = (grupo["mudanca_media_dp"] * grupo["n_celulas"]).sum() / n
            iguais = (grupo["pct_iguais"] * grupo["n_celulas"]).sum() / n

            ax.barh(y, mudanca, height=0.75, color=cores[metodo])

            # Ao lado da barra: a mudança e a % de células iguais, com vírgula
            texto = f"{mudanca:.3f} DP · {iguais:.1f}% iguais".replace(".", ",")
            ax.text(mudanca + 0.003, y, texto, va="center", fontsize=8.5, color=cor_secundaria)

            posicoes.append(y)
            posicoes_metodo.append(y)
            rotulos.append(f"{taxa:.0%}")
            y = y - 1

        meios.append(sum(posicoes_metodo) / len(posicoes_metodo))
        y = y - 0.9

    # 3. Nome de cada método à esquerda do grupo de barras dele
    #    (get_yaxis_transform: x em fração do painel e y nas posições das barras)
    for metodo, meio in zip(metodos, meios):
        ax.text(-0.13, meio, nomes_metodos[metodo], transform=ax.get_yaxis_transform(), ha="right",
                va="center", color=cor_texto, fontweight="bold")

    # Folga à direita para o texto da barra mais longa
    ax.set_xlim(0, ax.get_xlim()[1] * 1.5)
    ax.set_xlabel("mudança média do valor imputado (em DP da variável)", color=cor_eixos)
    ax.set_yticks(posicoes)
    ax.set_yticklabels(rotulos)
    ax.tick_params(colors=cor_eixos, length=0)
    ax.tick_params(axis="y", colors=cor_texto)
    ax.grid(axis="x", color=cor_grade, linewidth=0.8)
    ax.set_axisbelow(True)

    # Sem as bordas de cima, da direita e da esquerda
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.suptitle("Quanto o valor imputado muda quando os meses com valores extremos entram",
                 color=cor_texto, y=0.98)

    # rect deixa livre a faixa da esquerda (nomes dos métodos) e a de cima (título)
    plt.tight_layout(rect=(0.08, 0, 1, 0.95))

    # Salva sem fechar a figura, para ela ainda poder ser exibida
    if salvar:
        salvar_visualizacao(fig, nome)

    if plot:
        plt.show()
    else:
        plt.close(fig)


# %% [markdown]
# # Com × sem tratamento de outliers (modelos)

# %%
def halteres_r2(comparacao, nome="halteres_r2", plot=False, salvar=True):
    """
    Gera a figura do com × sem tratamento de outliers nos modelos: o R² de cada variável
    prevista na base tratada e na não tratada, lado a lado (halteres). Uma linha por
    modelo × variável; a barra liga os dois R², e a cor dela diz qual base teve o R² maior.

    Parameters
    ----------
    comparacao : pd.DataFrame
        Uma linha por modelo × métrica × variável, com as colunas "modelo", "metrica",
        "variavel", "tratada" e "nao_tratada" (a aba "comparacao" do
        ML_outliers_nao_tratados.xlsx). Só as linhas do R² entram na figura.
    nome : str, default "halteres_r2"
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
    """
    from matplotlib.lines import Line2D

    # Cores da barra: azul quando a tratada tem o R² maior, vermelho quando é a não tratada
    # (as mesmas da rodadas_ganhas_barras) e cinza se empatarem. Pontos em cinza escuro:
    # cheio = tratada; vazio = não tratada. Textos e eixos em tons neutros
    cores = {"tratada_melhor": "#2a78d6", "empate": "#c3c2b7", "nao_tratada_melhor": "#e34948"}
    nomes_modelos = {"rf": "RF", "knn": "KNN"}
    cor_pontos = "#52514e"
    cor_texto = "#0b0b0b"
    cor_secundaria = "#52514e"
    cor_eixos = "#898781"
    cor_grade = "#e1e0d9"

    # 1. Só as linhas do R²; modelos na ordem em que aparecem na tabela
    da_r2 = comparacao[comparacao["metrica"] == "r2"]
    modelos = list(da_r2["modelo"].unique())

    fig, ax = plt.subplots(figsize=(8, 8.6))

    # 2. Uma linha por modelo × variável, de cima para baixo: a posição y diminui a cada
    #    variável, e um espaço extra separa os modelos
    posicoes = []
    rotulos = []
    meios = []
    todos_r2 = []
    y = 0
    for modelo in modelos:
        do_modelo = da_r2[da_r2["modelo"] == modelo]
        posicoes_modelo = []

        for variavel in do_modelo["variavel"].unique():
            linha = do_modelo[do_modelo["variavel"] == variavel].iloc[0]
            r2_tratada = linha["tratada"]
            r2_nao_tratada = linha["nao_tratada"]

            # 3. Barra entre os dois R², com a cor da base que teve o R² maior
            if r2_tratada > r2_nao_tratada:
                cor_barra = cores["tratada_melhor"]
            elif r2_tratada < r2_nao_tratada:
                cor_barra = cores["nao_tratada_melhor"]
            else:
                cor_barra = cores["empate"]
            ax.plot([r2_tratada, r2_nao_tratada], [y, y], color=cor_barra, linewidth=3,
                    solid_capstyle="round", zorder=2)

            # 4. Pontos: a tratada cheia e a não tratada vazia, por cima (zorder maior)
            ax.scatter(r2_tratada, y, s=58, color=cor_pontos, edgecolor="white", linewidth=0.8, zorder=3)
            ax.scatter(r2_nao_tratada, y, s=58, facecolor="white", edgecolor=cor_pontos, linewidth=1.6,
                       zorder=4)

            posicoes.append(y)
            posicoes_modelo.append(y)
            rotulos.append(variavel)
            todos_r2.append(r2_tratada)
            todos_r2.append(r2_nao_tratada)
            y = y - 1

        meios.append(sum(posicoes_modelo) / len(posicoes_modelo))
        y = y - 0.9

    # 5. Linha do R² = 0 (prever sempre a média dá R² = 0), com uma nota embaixo
    ax.axvline(0, color=cor_secundaria, linewidth=1, linestyle="--", zorder=1)
    ax.text(0.005, posicoes[-1] - 1, "R² = 0: igual a prever com a média", color=cor_eixos, fontsize=8,
            va="center")

    # 6. Limites do eixo x: do menor ao maior R² (com o 0 sempre dentro), com uma folga de 5%
    menor = min(min(todos_r2), 0)
    maior = max(max(todos_r2), 0)
    folga = 0.05 * (maior - menor)
    ax.set_xlim(menor - folga, maior + folga)
    ax.set_ylim(posicoes[-1] - 1.5, posicoes[0] + 0.7)

    # 7. Nome de cada modelo à esquerda do grupo dele
    #    (get_yaxis_transform: x em fração do painel e y nas posições das linhas)
    for modelo, meio in zip(modelos, meios):
        ax.text(-0.2, meio, nomes_modelos[modelo], transform=ax.get_yaxis_transform(), ha="right",
                va="center", color=cor_texto, fontweight="bold")

    ax.set_xlabel("R²", color=cor_eixos)
    ax.set_yticks(posicoes)
    ax.set_yticklabels(rotulos)
    ax.tick_params(colors=cor_eixos, length=0)
    ax.tick_params(axis="y", colors=cor_texto)
    ax.grid(axis="x", color=cor_grade, linewidth=0.8)
    ax.set_axisbelow(True)

    # Sem as bordas de cima, da direita e da esquerda
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # 8. Uma legenda para a figura toda. As linhas e os pontos (Line2D) só desenham os símbolos
    simbolos = [
        Line2D([], [], marker="o", linestyle="", markersize=8, color=cor_pontos),
        Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor="white",
               markeredgecolor=cor_pontos, markeredgewidth=1.6),
        Line2D([], [], color=cores["tratada_melhor"], linewidth=3),
        Line2D([], [], color=cores["nao_tratada_melhor"], linewidth=3),
    ]
    rotulos_legenda = ["tratada", "não tratada", "R² maior na tratada", "R² maior na não tratada"]
    fig.suptitle("Com × sem tratamento de outliers nos modelos: o R² em cada base", color=cor_texto, y=0.995)
    fig.legend(simbolos, rotulos_legenda, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2,
               frameon=False)
    fig.text(0.5, 0.885, "RF: Random Forest, alvo pH · KNN: cada variável prevista pelas outras · "
             "validação cruzada em 5 blocos de meses seguidos", ha="center", color=cor_secundaria, fontsize=8)

    # rect deixa livre a faixa da esquerda (nomes dos modelos) e a de cima (título e legenda)
    plt.tight_layout(rect=(0.06, 0, 1, 0.87))

    # Salva sem fechar a figura, para ela ainda poder ser exibida
    if salvar:
        salvar_visualizacao(fig, nome)

    if plot:
        plt.show()
    else:
        plt.close(fig)


# %% [markdown]
# # Comparação entre bases (descritiva evolutiva)

# %%
def boxplots_bases(bases, variaveis=None, nome="boxplots_bases", plot=False, salvar=True):
    """
    Gera os boxplots de cada variável em várias bases, lado a lado: um painel por
    variável, com uma caixa por base (na ordem do dicionário). Serve para comparar
    a distribuição de uma variável antes e depois de uma etapa (ex.: limpeza,
    tratamento de outliers, imputação).

    Os pontos soltos são os valores além de 1,5 × IQR (a regra do próprio boxplot),
    e não os outliers de algum método de detecção.

    Parameters
    ----------
    bases : dict
        Nome de cada base (rótulo da legenda) → pd.DataFrame com as variáveis em colunas.
    variaveis : list, optional
        Variáveis desenhadas, uma por painel. Se None, usa as variáveis da primeira
        base (separar_colunas), by default None
    nome : str, default "boxplots_bases"
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
    """
    # Cores das bases, na ordem do dicionário (a primeira em cinza: em geral, a base de partida);
    # textos e eixos em tons neutros
    cores = ["#898781", "#2a78d6", "#1baf7a", "#eb6834", "#6250d6", "#e34948"]
    cor_texto = "#0b0b0b"
    cor_secundaria = "#52514e"
    cor_eixos = "#898781"
    cor_grade = "#e1e0d9"

    # 1. Nomes das bases e variáveis (as da primeira base, se não forem dadas)
    nomes_bases = list(bases.keys())
    if variaveis is None:
        from pcj.utils import separar_colunas
        _, variaveis = separar_colunas(bases[nomes_bases[0]])

    # 2. Grade de painéis: 4 colunas e quantas linhas forem precisas
    #    (squeeze=False: axs é sempre uma tabela de painéis, mesmo com uma linha só)
    n_colunas = 4
    n_linhas = math.ceil(len(variaveis) / n_colunas)
    fig, axs = plt.subplots(n_linhas, n_colunas, figsize=(14, 3 * n_linhas), squeeze=False)

    for n, variavel in enumerate(variaveis):
        # Linha e coluna do painel desta variável (// é a divisão inteira; % é o resto)
        ax = axs[n // n_colunas, n % n_colunas]

        # 3. Uma caixa por base: os valores da variável, sem os NaN (o boxplot não aceita NaN)
        valores = []
        for nome_base in nomes_bases:
            valores.append(bases[nome_base][variavel].dropna())

        caixas = ax.boxplot(valores, widths=0.6, patch_artist=True,
                            medianprops={"color": cor_texto, "linewidth": 1.2},
                            whiskerprops={"color": cor_secundaria}, capprops={"color": cor_secundaria},
                            flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "none",
                                        "markeredgecolor": cor_secundaria, "alpha": 0.6})

        # Cor de cada caixa (patch_artist=True permite pintar o interior)
        for b, caixa in enumerate(caixas["boxes"]):
            caixa.set_facecolor(cores[b % len(cores)])
            caixa.set_edgecolor(cor_secundaria)

        # Título, eixos e grade. As bases não aparecem no eixo x: a cor diz qual é (ver a legenda)
        ax.set_title(variavel, color=cor_texto)
        ax.set_xticks([])
        ax.tick_params(colors=cor_eixos)
        ax.grid(axis="y", color=cor_grade, linewidth=0.8)
        ax.set_axisbelow(True)

        # Sem as bordas de cima e da direita
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 4. Painéis que sobram na grade ficam em branco
    for n in range(len(variaveis), n_linhas * n_colunas):
        axs[n // n_colunas, n % n_colunas].axis("off")

    # 5. Uma legenda para a figura toda. Os quadrados (Rectangle) só desenham as cores
    quadrados = []
    for b in range(len(nomes_bases)):
        quadrados.append(plt.Rectangle((0, 0), 1, 1, color=cores[b % len(cores)]))
    fig.suptitle("Distribuição de cada variável em cada base", color=cor_texto, y=0.995)
    fig.legend(quadrados, nomes_bases, loc="upper center", bbox_to_anchor=(0.5, 0.97),
               ncol=len(nomes_bases), frameon=False)
    fig.text(0.5, 0.935, "Pontos soltos: valores além de 1,5 × IQR (a regra do boxplot)", ha="center",
             color=cor_secundaria, fontsize=9)

    # rect deixa livre a faixa de cima da figura, onde ficam o título e a legenda
    plt.tight_layout(rect=(0, 0, 1, 0.93))

    # Salva sem fechar a figura, para ela ainda poder ser exibida
    if salvar:
        salvar_visualizacao(fig, nome)

    if plot:
        plt.show()
    else:
        plt.close(fig)


# %%
def series_marcadas(base, marcas=None, variaveis=None, coluna_data="Data", nome="series_marcadas",
                    plot=False, salvar=True):
    """
    Gera a série temporal de cada variável de uma base (um painel por variável) e marca
    por cima conjuntos de valores escolhidos (ex.: os valores tratados como outliers, os
    valores imputados). Os meses sem valor na base aparecem como falhas na linha.

    Parameters
    ----------
    base : pd.DataFrame
        Base desenhada em linha, com a coluna de datas e as variáveis.
    marcas : dict, optional
        Rótulo de cada conjunto (legenda) → pd.DataFrame com a coluna de datas e as
        variáveis, com NaN em tudo o que não deve ser marcado. Cada conjunto ganha uma
        cor e um símbolo. Se None, só desenha as séries, by default None
    variaveis : list, optional
        Variáveis desenhadas, uma por painel. Se None, usa as variáveis da base
        (separar_colunas), by default None
    coluna_data : str, default "Data"
        Coluna com as datas (eixo x).
    nome : str, default "series_marcadas"
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
    """
    from matplotlib.lines import Line2D

    # Cores e símbolos dos conjuntos marcados, na ordem do dicionário; a linha da base em cinza escuro
    cores = ["#e34948", "#1baf7a", "#eb6834", "#6250d6", "#2a78d6"]
    simbolos = ["x", "o", "s", "^", "D"]
    cor_linha = "#52514e"
    cor_texto = "#0b0b0b"
    cor_eixos = "#898781"
    cor_grade = "#e1e0d9"

    if marcas is None:
        marcas = {}

    # 1. Variáveis da base, se não forem dadas
    if variaveis is None:
        from pcj.utils import separar_colunas
        _, variaveis = separar_colunas(base)

    # 2. Grade de painéis: 4 colunas e quantas linhas forem precisas
    n_colunas = 4
    n_linhas = math.ceil(len(variaveis) / n_colunas)
    fig, axs = plt.subplots(n_linhas, n_colunas, figsize=(16, 2.8 * n_linhas), squeeze=False, sharex=True)

    for n, variavel in enumerate(variaveis):
        ax = axs[n // n_colunas, n % n_colunas]

        # 3. A série da base: a linha se interrompe nos meses sem valor (NaN)
        ax.plot(base[coluna_data], base[variavel], color=cor_linha, linewidth=0.9, zorder=2)

        # 4. Cada conjunto marcado: só os valores que não são NaN
        for m, rotulo in enumerate(marcas):
            conjunto = marcas[rotulo]
            com_valor = conjunto[variavel].notna()
            ax.scatter(conjunto.loc[com_valor, coluna_data], conjunto.loc[com_valor, variavel],
                       marker=simbolos[m % len(simbolos)], s=22, color=cores[m % len(cores)],
                       linewidths=1.2, zorder=3 + m)

        ax.set_title(variavel, color=cor_texto)
        ax.tick_params(colors=cor_eixos, labelsize=8)
        ax.grid(axis="y", color=cor_grade, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 5. Painéis que sobram na grade ficam em branco
    for n in range(len(variaveis), n_linhas * n_colunas):
        axs[n // n_colunas, n % n_colunas].axis("off")

    # 6. Uma legenda para a figura toda: a linha da base e os símbolos de cada conjunto
    simbolos_legenda = [Line2D([], [], color=cor_linha, linewidth=1.5)]
    rotulos_legenda = ["série da base (falhas = meses sem valor)"]
    for m, rotulo in enumerate(marcas):
        simbolos_legenda.append(Line2D([], [], marker=simbolos[m % len(simbolos)], linestyle="",
                                       markersize=6, color=cores[m % len(cores)], markeredgewidth=1.2))
        rotulos_legenda.append(rotulo)
    fig.suptitle("Série de cada variável, com os valores marcados", color=cor_texto, y=0.995)
    fig.legend(simbolos_legenda, rotulos_legenda, loc="upper center", bbox_to_anchor=(0.5, 0.97),
               ncol=len(rotulos_legenda), frameon=False)

    plt.tight_layout(rect=(0, 0, 1, 0.94))

    # Salva sem fechar a figura, para ela ainda poder ser exibida
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
