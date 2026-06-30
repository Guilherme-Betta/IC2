
# CLAUDE.md — Projeto IC2

Projeto de Iniciação Científica. Contexto técnico para o Claude Code.

## Stack

Python 3.9+, pandas, numpy, scipy, scikit-learn, statsmodels,
matplotlib, seaborn, missingno, openpyxl, Jupyter

## Estrutura

- src/pcj/        → módulos Python reutilizáveis (processamento, outliers,
  imputação, visualizações, etc.)
- notebooks/      → pipeline principal (numerado 0–6) e tutoriais de ML
- data/raw/       → dados brutos originais (não modificar)
- data/processed/ → dados em estágios intermediários
- data/output/    → saídas geradas automaticamente (figuras, métricas,
  com timestamp)
- docs/           → documentação e dicionário de dados

## Comandos comuns

- Abrir ambiente: jupyter notebook
- Testes: pytest  (a pasta tests/ existe mas ainda está vazia)
- Instalar dependências: pip install -e .
  (lê o pyproject.toml; requirements.txt é gerado via pip freeze
  apenas na entrega final do projeto)

## Convenções

- Nomes de funções e variáveis em português (ex: dados_limpos,
  metadados, variaveis, salvar)
- Salvar saídas sempre via salvar() em src/pcj/utils.py — ela adiciona
  timestamp e grava em data/output/ automaticamente
- Figuras salvas via salvar_visualizacao() em data/output/figures/

## Regras de dados

- NUNCA sobrescreva arquivos em data/raw/ — é a fonte original e
  imutável dos dados. Toda transformação gera saída em data/processed/
  ou data/output/.

