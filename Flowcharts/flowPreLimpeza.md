```mermaid
graph TD
    A["Início: Arquivo 'dadosLargos.xlsx'"] --> B{"Ler Planilhas"};
    B -->|"Planilhas '2009'-'2024'"| C["Loop: Processar Anos"];
    B -->|"Planilha 'DadosAnuais'"| D["Processar Separado"];
    
    subgraph "Tratamento Mensal (Loop)"
    C --> C1["Ler Aba"];
    C1 --> C2["Converter Datas 'pt-BR'"];
    C2 --> C3{"Tem Coluna Extra?"};
    C3 -->|Sim| C4["Criar MultiIndex (Data, Estatística)"];
    C3 -->|Não| C5["Index Simples (Data)"];
    C4 & C5 --> C6["Remover Colunas Vazias"];
    end
    
    C6 --> E["Concatenar Tudo (pd.concat)"];
    E --> F["Identificar Variáveis: Core vs Tardias"];
    
    subgraph "Geração de Saída (Loop)"
    F --> G1["Salvar 'dados_core'"];
    F --> G2["Iterar Datas de Início das Tardias"];
    G2 --> G3["Filtrar Dados >= Data Início"];
    G3 --> G4["Gerar Nome Descritivo"];
    G4 --> G5["Salvar Aba Incremental"];
    end
    
    D --> H["Limpar e Definir Índice 'Ano'"];
    H --> I["Salvar Aba 'dados_anuais'"];
    
    G1 & G5 & I --> J["FIM: 'dados_analiticos_graduais.xlsx'"];
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style J fill:#9f9,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333
 ```