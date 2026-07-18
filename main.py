import pandas as pd
from src.eda import executar_eda

if __name__ == "__main__":
    # Caminho relativo para os dados
    caminho_dados = "data/manutencao_preditiva.csv"
    
    print("[INFO] Iniciando o Pipeline Preditivo Industrial...")
    
    # Carregando o arquivo CSV direto utilizando o Pandas
    df = pd.read_csv(caminho_dados)
    print(f"[SUCESSO] Base de dados carregada de: {caminho_dados}\n")
    
    # --- EXECUÇÃO DA FASE 1 ---
    executar_eda(df)
    
    print("\n[SUCESSO] Fase 1 concluída! Os gráficos foram salvos na pasta /data.")
    print("Próximo passo: Implementar e chamar a Fase 2 de Data Prep.")
