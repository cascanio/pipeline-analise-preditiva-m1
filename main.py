import pandas as pd
from src.eda import executar_eda
from src.data_prep import preparar_dados

if __name__ == "__main__":
    caminho_dados = "data/manutencao_preditiva.csv"
    
    print("[INFO] Iniciando o Pipeline Preditivo Industrial...")
    
    # Carregando os dados
    df = pd.read_csv(caminho_dados)
    print(f"[SUCESSO] Base de dados carregada de: {caminho_dados}\n")
    
    # Executando a Fase 1 (EDA)
    executar_eda(df)
    
    # Executando as Fases 2, 3, 4 e 5 (Preparação e Limpeza)
    X_train_knn, X_test_knn, X_train_tree, X_test_tree, y_train, y_test = preparar_dados(df)
    
    print("\n[SUCESSO] Fases de preparação concluídas perfeitamente!")
    print("Próximo passo: Implementar a Fase 6 e 7 de Modelagem.")
