Todo .py que é só definição de funções (o que você exportou dos notebooks sem # Teste).

Exemplo de layout:
~~~
src/
  projeto_ic/          # nome do pacote (pode ser curto)
    __init__.py
    utils.py
    processamento.py
    outliers.py
    imputacao.py
Notebooks importam assim (depois de configurar o pacote — ver imports abaixo):
~~~

from projeto_ic.imputacao import mean_imput