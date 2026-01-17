```mermaid
flowchart TD
    %% Nós Iniciais
    Start([Início: Arquivo 'dadosLargos.xlsx']) --> CheckType{Ler Planilhas}
    
    %% Ramificações
    CheckType -->|Planilhas '2009'-'2024'| LoopMensal[Loop: Processar Anos]
    CheckType -->|Planilha 'DadosAnuais'| ProcAnual[Processar Separado]
    
    %% Subgráfico: Tratamento Mensal
    subgraph Tratamento [Tratamento Mensal - Função ler_e_limpar]
        direction TB
        LoopMensal --> ReadSheet[Ler Aba]
        ReadSheet --> ConvertDate[Converter Datas 'pt-BR']
        ConvertDate --> HasExtra{Tem Coluna<br/>Extra?}
        
        HasExtra -- Sim --> MultiIdx[MultiIndex<br/>Data + Estatística]
        HasExtra -- Não --> SingleIdx[Index Simples<br/>Data]
        
        MultiIdx & SingleIdx --> DropEmpty[Remover Colunas Vazias]
    end
    
    DropEmpty --> Concat[Concatenar Tudo<br/>pd.concat]
    Concat --> IdentVars[Identificar Variáveis:<br/>Core vs Tardias]
    
    %% Subgráfico: Geração de Saída
    subgraph Saida [Geração de Saída - Loop]
        direction TB
        IdentVars --> SaveCore[Salvar aba 'dados_core']
        IdentVars --> IterDates[Iterar Datas de<br/>Início das Tardias]
        IterDates --> FilterData[Filtrar Dados >= Data Início]
        FilterData --> GenName[Gerar Nome Descritivo]
        GenName --> SaveInc[Salvar Aba Incremental]
    end
    
    %% Caminho Anual
    ProcAnual --> CleanAnual[Limpar e Definir Índice 'Ano']
    CleanAnual --> SaveAnual[Salvar Aba 'dados_anuais']
    
    %% Finalização
    SaveCore & SaveInc & SaveAnual --> End([FIM: Arquivo com Timestamp])
    
    %% Estilização (Opcional - Cores suaves)
    style Start fill:#2c3e50,stroke:#333,color:#fff
    style End fill:#2c3e50,stroke:#333,color:#fff
    style Concat fill:#e67e22,stroke:#333,color:#fff
    style IdentVars fill:#e67e22,stroke:#333,color:#fff
 ```