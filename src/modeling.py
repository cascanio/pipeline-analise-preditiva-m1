from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def executar_modelagem(X_train_knn, X_test_knn, X_train_tree, X_test_tree, y_train, y_test):
    """
    Objetivo: Treinar os modelos KNN e Árvore variando os parâmetros de controle.
    Para que: Combater o overfitting e extrair as acurácias finais (Critérios 9 e 10).
    """
    print("\n--- FASE 6: AJUSTE DE PARÂMETROS E COMBATE AO OVERFITTING ---")
    
    # 🧪 Experimentação 1: KNN (Variando K por 3 valores ímpares)
    print("\n[Treinando KNN...]")
    valores_k = [3, 5, 7]
    melhor_acc_knn = 0
    melhor_k = 3
    
    for k in valores_k:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_knn, y_train)
        
        # Registrando acurácias de treino e teste para verificar overfitting
        acc_treino = accuracy_score(y_train, knn.predict(X_train_knn))
        acc_teste = accuracy_score(y_test, knn.predict(X_test_knn))
        print(f"KNN (K={k}) -> Acurácia Treino: {acc_treino:.4f} | Teste: {acc_teste:.4f}")
        
        if acc_teste > melhor_acc_knn:
            melhor_acc_knn = acc_teste
            melhor_k = k

    # 🧪 Experimentação 2: Árvore de Decisão (Variando profundidade máxima)
    print("\n[Treinando Árvore de Decisão...]")
    profundidades = [3, 5, None]
    melhor_acc_tree = 0
    melhor_depth = 3
    
    for d in profundidades:
        tree = DecisionTreeClassifier(max_depth=d, random_state=42)
        tree.fit(X_train_tree, y_train)
        
        # Registrando acurácias de treino e teste
        acc_treino = accuracy_score(y_train, tree.predict(X_train_tree))
        acc_teste = accuracy_score(y_test, tree.predict(X_test_tree))
        print(f"Árvore (max_depth={d}) -> Acurácia Treino: {acc_treino:.4f} | Teste: {acc_teste:.4f}")
        
        if acc_teste > melhor_acc_tree:
            melhor_acc_tree = acc_teste
            melhor_depth = d

    print("\n--- FASE 7: AVALIAÇÃO DA ACURÁCIA E VEREDITO FINAL ---")
    print(f"Melhor acurácia obtida pelo KNN (K={melhor_k}): {melhor_acc_knn:.4f}")
    print(f"Melhor acurácia obtida pela Árvore (max_depth={melhor_depth}): {melhor_acc_tree:.4f}")
    
    # Decisão automática do modelo vencedor com base na acurácia de teste
    if melhor_acc_knn > melhor_acc_tree:
        print(f"\nVEREDITO: O modelo KNN superou a Árvore no teste ({melhor_acc_knn:.4f}) e deve ser adotado.")
    else:
        print(f"\nVEREDITO: O modelo Árvore de Decisão superou o KNN no teste ({melhor_acc_tree:.4f}) e deve ser adotado.")
