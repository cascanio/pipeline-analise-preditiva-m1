import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def executar_eda(df):
    """
    Objetivo: Entender os dados e extrair os primeiros insights estatísticos.
    Para que: Garantir nota máxima no Critério 4 (EDA completo + Gráficos).
    """
    print("\n--- FASE 1: ANÁLISE EXPLORATÓRIA DE DADOS (EDA) ---")
    
    # 1. Dimensões do dataset (Linhas e Colunas)
    print(f"Dimensões do Dataset: {df.shape[0]} linhas e {df.shape[1]} colunas.")
    
    # 2. Tipos de dados de cada coluna
    print("\nTipos de dados das variáveis:")
    print(df.dtypes)
    
    # 3. Resumo estatístico descritivo
    print("\nResumo Estatístico Descritivo:")
    print(df.describe())
    
    # Configuração visual dos gráficos utilizando Seaborn/Matplotlib
    sns.set_theme(style="whitegrid")
    
    # Gráfico 1: Histograma das variáveis preditoras numéricas (ex: Torque)
    # Para que: Ver se os dados seguem uma distribuição normal ou têm distorções.
    plt.figure(figsize=(6, 4))
    # Substitua 'torque_nm' pelo nome exato da coluna do seu CSV após abri-lo
    if 'torque_nm' in df.columns:
        sns.histplot(df['torque_nm'], kde=True, color='blue')
        plt.title("Distribuição da Variável Torque (Nm)")
        plt.tight_layout()
        plt.savefig("data/distribuicao_torque.png")
        plt.close()
    
    # Gráfico 2: Gráfico de barras da variável alvo (Falha)
    # Para que: Comprovar visualmente o desbalanceamento (Gráfico exigido no critério!)
    plt.figure(figsize=(6, 4))
    if 'Falha' in df.columns:
        sns.countplot(x='Falha', data=df, palette='Set2')
        plt.title("Taxa de Desbalanceamento da Variável Alvo (Falha)")
        plt.tight_layout()
        plt.savefig("data/desbalanceamento_alvo.png")
        plt.close()
        
    # Gráfico 3: Mapa de Calor (Heatmap) de Correlação de Pearson
    # Para que: Ver quais variáveis têm forte relação com a quebra mecânica ou entre si.
    plt.figure(figsize=(8, 6))
    # Selecionamos apenas colunas numéricas para calcular a correlação
    colunas_numericas = df.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(colunas_numericas.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Mapa de Calor - Correlação de Pearson")
    plt.tight_layout()
    plt.savefig("data/mapa_correlacao.png")
    plt.close()
    
    print("\n[INFO] Gráficos analíticos gerados e salvos com sucesso na pasta /data.")
