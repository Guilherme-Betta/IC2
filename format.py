import pandas as pd

_casas_decimais_padrao = None

def set_casas_decimais_padrao(n_casas_decimais):
    """
    Determina o número padrão de casas decimais para cada variável.
    """
    global _casas_decimais_padrao
    _casas_decimais_padrao = n_casas_decimais

def get_casas_decimais_padrao():
    """Retorna o número padrão de casas decimais"""
    return _casas_decimais_padrao

def casas_decimais(coluna):
    """
    Funcao que determina a quantidade de casas decimais utilizadas em UMA coluna
    de um banco de dados.
    Serah utilizada para padronizar o numero de casas decimais por variavel, com base
    no numero de casas decimais utilizadas no banco original.
    """
    # Elimina celulas vazias e converte todos os valores da coluna para string
    textos = coluna.dropna().astype(str)    

    # Busca celulas que contem "." 
    decimais = textos[textos.str.contains(r'\.', regex=True, na=False)] 

    if decimais.empty:
        return 0    # Colunas sem "." sao definidas como contendo 0 casas decimais
    
    n_decimais = decimais.str.split('.').str[1].str.len()

    # Retorna o valor máximo de dígitos após a casa decimal como integer
    return int(n_decimais.max())

def formatar_variaveis(describe, n_casas_decimais):
    """
    Formata o DataFrame describe com casas decimais baseadas nos dados originais.
    describe: DataFrame de estatísticas.
    dados: DataFrame original.
    variaveis: Lista de colunas variáveis.
    """

    # Selecao de estatisticas cujos valores devem ser formatados. 
    # Exclui as estatisticas criadas e adicionadas a tabela,
    # pois essas sao porcentagens ou contagens (numeros inteiros), que serao 
    # padronizadas com 2 e 0 casas decimais respectivamente
    estatisticas_para_formatar = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']

    # Loop para iterar sobre as variaveis numericas
    for i in n_casas_decimais:

        # Regra de formatacao com base nos dados armazenados em n_casas_decimais (ex: se casas=2, vira "{:.2f}")
        regra_formatacao = f"{{:.{n_casas_decimais[i]}f}}"

        # Aplicacao da formatacao nas estatísticas selecionadas no primeiro passo
        dados_matematicos = describe.loc[estatisticas_para_formatar, i]
        describe.loc[estatisticas_para_formatar, i] = dados_matematicos.apply(
            lambda x: regra_formatacao.format(float(x)) if pd.notna(x) else x
        )
    return describe

def formatar_metadados(describe, metadados):
    """
    Limpa o DataFrame describe, removendo estatísticas desnecessárias para metadados.
    describe: DataFrame de estatísticas.
    metadados: Lista de colunas metadados.
    estatisticas_manter: Estatísticas a manter para metadados.
    transpose: Permite selecionar se o quadro de estatísticas deve ser pivotado ou não (Pivotado by default).
    """
    # Lista de estatisticas de interesse para as colunas de metadados
    estatisticas_manter = ['count', 'unique', 'freq']

    # Cria uma lista das estatisticas que nao estao na lista do passo anterior
    estatisticas_apagar = [stat for stat in describe.index if stat not in estatisticas_manter]

    # Elimina as estatisticas que nao estao na lista do primeiro passo
    describe.loc[estatisticas_apagar, metadados] = ''

    # Retorna o quadro de estatísticas
    return describe