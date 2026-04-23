from utils import*
import pandas as pd
# import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns

dados_limpos = dados(r"data\out\3) Dados Limpos\dados_limpos20260420_161355.xlsx")

iqr_knn = dados(r"data\out\5) Imputação\IQR\KNN\imput_knn20260420_163032.xlsx")

iqr_media = dados(r"data\out\5) Imputação\IQR\Média\imput_mean20260420_163032.xlsx")

iqr_mediana = dados(r"data\out\5) Imputação\IQR\Mediana\imput_median20260420_163032.xlsx")

zscore_knn = dados(r"data\out\5) Imputação\Z-Score\KNN\imput_knn20260420_163302.xlsx")

zscore_media = dados(r"data\out\5) Imputação\Z-Score\Média\imput_mean20260420_163302.xlsx")

zscore_mediana = dados(r"data\out\5) Imputação\Z-Score\Mediana\imput_median20260420_163302.xlsx")

dados_limpos.plot(
    kind='bar') 