import numpy as np
from knn import KNN
from naive_bayes import NaiveBayes
from decision_tree import DecisionTree
from utils import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

# Carregar o conjunto de dados
iris = datasets.load_iris()

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25)

# Treinar os algoritmos
knn = KNN(k=3)
knn.fit(X_train, y_train)

nb = NaiveBayes()
nb.fit(X_train, y_train)

dt = DecisionTree()
dt.fit(X_train, y_train)

# Fazer previsões para o conjunto de teste
y_pred_knn = knn.predict(X_test)
y_pred_nb = nb.predict(X_test)
y_pred_dt = dt.predict(X_test)

# Avaliar os resultados
for clf, y_pred in zip([knn, nb, dt], [y_pred_knn, y_pred_nb, y_pred_dt]):
    print(f"--- {clf.__class__.__name__} ---")
    print(f"Acurácia: {accuracy_score(y_test, y_pred)}")
    print(f"Matriz de Confusão: {confusion_matrix(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"Precisão: {precision_score(y_test, y_pred)}")
    print(f"F1-Score: {f1_score(y_test, y_pred)}")