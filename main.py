import pandas as pd
from src.eda import executar_eda
from src.data_prep import preparar_dados
from src.modeling import executar_modelagem

if __name__ == "__main__":
    caminho_dados = "data/manutencao_preditiva.csv"
    
    print("[INFO] Iniciando o Pipeline Preditivo Industrial...")
    
    # Carregando os dados
    df = pd.read_csv(caminho_dados)
    print(f"[SUCESSO] Base de dados carregada de: {caminho_dados}\n")
    
    # Executando o Pipeline Completo Passo a Passo
    executar_eda(df)
    
    X_train_knn, X_test_knn, X_train_tree, X_test_tree, y_train, y_test = preparar_dados(df)
    
    executar_modelagem(X_train_knn, X_test_knn, X_train_tree, X_test_tree, y_train, y_test)
    
    print("\n[SUCESSO] Pipeline completo executado com sucesso absoluto!")
