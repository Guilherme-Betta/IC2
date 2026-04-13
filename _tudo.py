from utils import *
from analise_descritiva import *
from imputacao import *
from limpeza import *
from outliers import *
from visualizacoes import *

# Carregar os dados brutos
dados_brutos, metadados, variaveis = dados(r"data\in\dados_brutos.xlsx")

# Primeira análise descritiva e conjunto de visualizações
analise_descritiva(dados_brutos, metadados=metadados, variaveis=variaveis)
visualizacoes_todas(dados_brutos)

# Dados brutos prontos
dados_prontos = prep_brutos(dados_brutos)

analise_descritiva(dados_prontos, metadados=metadados, variaveis=variaveis)
visualizacoes_todas(dados_prontos)

# Dados limpos
dados_limpos = limpeza(dados_prontos, metadados=metadados, variaveis=variaveis)

analise_descritiva(dados_limpos, metadados=metadados, variaveis=variaveis)
visualizacoes_todas(dados_limpos)

# Outliers

## Z-Score
remocao_z_score = remove_outliers(dados_limpos, variaveis=variaveis, method='zscore')

analise_descritiva(remocao_z_score, metadados=metadados, variaveis=variaveis)
visualizacoes_todas(remocao_z_score)

## IQR
remocao_iqr = remove_outliers(dados_limpos, variaveis=variaveis, method='iqr')

analise_descritiva(remocao_iqr, metadados=metadados, variaveis=variaveis)
visualizacoes_todas(remocao_iqr)

## LOF
remocao_lof = remove_outliers(dados_limpos, variaveis=variaveis, method='lof')

analise_descritiva(remocao_lof, metadados=metadados, variaveis=variaveis)
visualizacoes_todas(remocao_lof)

# Imputação

## Média
imput_media = mean_imput(dados_limpos, variaveis=variaveis)

analise_descritiva(imput_media, metadados=metadados, variaveis=variaveis)
visualizacoes_todas(imput_media)

## Mediana
imput_mediana = median_imput(dados_limpos, variaveis=variaveis)

analise_descritiva(imput_mediana, metadados=metadados, variaveis=variaveis)
visualizacoes_todas(imput_mediana)

## KNN
imput_KNN = knn_imput(dados_limpos, variaveis=variaveis)

analise_descritiva(imput_KNN, metadados=metadados, variaveis=variaveis)
visualizacoes_todas(imput_KNN)