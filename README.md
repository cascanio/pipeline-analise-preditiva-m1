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

### 📊 Justificativas Técnicas & Análise de Overfitting
- **Imputação por Mediana**: Utilizada na Fase 2 porque a base de dados industriais apresenta variações bruscas (outliers). A média seria distorcida por esses valores extremos, enquanto a mediana mantém o centro estável.
- **Escalonamento**: Aplicado via `StandardScaler` apenas para o modelo KNN. A Árvore de Decisão é imune à escala dos atributos porque cria quebras perpendiculares isoladas nos eixos de decisão.
- **Overfitting**: Observado na Árvore de Decisão quando `max_depth=None`. O modelo atingiu acurácia máxima no treino, mas reduziu seu desempenho no teste por memorizar os dados. A estabilidade foi alcançada aplicando a poda do modelo.
- **Veredito**: O modelo de **Árvore de Decisão** superou o KNN nos dados de teste, alcançando a acurácia final de **0.9385**, sendo o modelo escolhido para adoção pela empresa.
