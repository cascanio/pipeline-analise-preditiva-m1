# 🔧 FactoryPredict: Pipeline de Manutenção Preditiva

### 📝 Descrição do Problema
O objetivo deste projeto é prever quebras mecânicas em equipamentos industriais monitorados por sensores, permitindo intervenções planejadas antes que ocorram paradas inesperadas na linha de produção (Indústria 4.0).

### 🛠️ Técnicas e Tecnologias Utilizadas
- **Linguagem**: Python 3
- **Ambiente Virtual**: venv
- **Bibliotecas**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn e Imbalanced-Learn (SMOTE).
- **Modelos**: KNN e Árvore de Decisão.

### 🚀 Como Executar o Sistema
1. Clone o repositório: `git clone git@github.com:cascanio/pipeline-analise-preditiva-m1.git`
2. Crie e ative o ambiente virtual: `python -m venv venv` e ative de acordo com seu SO.
3. Instale as dependências: `pip install -r requirements.txt`
4. Execute o pipeline: `python main.py`

### 📋 Anotações do Departamento de Engenharia
Para garantir a reprodutibilidade do modelo e respeitar as regras físicas dos equipamentos, as seguintes diretrizes foram aplicadas no desenvolvimento do pipeline:

1. **Variável Alvo**: Definida como `falha_maquina` (0 = funcionamento normal, 1 = avaria detectada).
2. **Exclusão de Atributos**: Os motivos específicos de quebra (`falha_twf`, `falha_hdf`, `falha_pwf`, `falha_osf`, `falha_rnf`) foram descartados do treinamento. Mantê-los causaria *Data Leakage* (vazamento de dados), pois na vida real o sensor não sabe o motivo da quebra antes dela acontecer.
3. **Mapeamento de Variáveis Utilizadas**:
   - `temperatura_ar_k`: Temperatura ambiente medida em Kelvin.
   - `temperatura_processo_k`: Temperatura de operação do processo em Kelvin.
   - `velocidade_rotacao_rpm`: Velocidade do eixo calculada em RPM.

### 📊 Justificativas Técnicas & Análise de Overfitting
- **Imputação por Mediana**: Utilizada na Fase 2 porque a base de dados industriais apresenta variações bruscas (outliers). A média seria distorcida por esses valores extremos, enquanto a mediana mantém o centro estável.
- **Escalonamento**: Aplicado via `StandardScaler` apenas para o modelo KNN. A Árvore de Decisão é imune à escala dos atributos porque cria quebras perpendiculares isoladas nos eixos de decisão.
- **Overfitting**: Observado na Árvore de Decisão quando `max_depth=None`. O modelo atingiu acurácia máxima no treino, mas reduziu seu desempenho no teste por memorizar os dados. A estabilidade foi alcançada aplicando a poda do modelo.
- **Veredito**: O modelo de **Árvore de Decisão** superou o KNN nos dados de teste, alcançando a acurácia final de **0.9385**, sendo o modelo escolhido para adoção pela empresa.
