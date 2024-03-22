def accuracy_score(y_true, y_pred):
  """
  Calcula a acurácia do modelo.

  Parâmetros:
    y_true: array-like, rótulos verdadeiros.
    y_pred: array-like, rótulos preditos.

  Retorna:
    Acurácia do modelo.
  """
  correct = np.sum(y_true == y_pred)
  total = len(y_true)
  return correct / total

def confusion_matrix(y_true, y_pred):
  """
  Calcula a matriz de confusão do modelo.

  Parâmetros:
    y_true: array-like, rótulos verdadeiros.
    y_pred: array-like, rótulos preditos.

  Retorna:
    Matriz de confusão do modelo.
  """
  unique_labels = np.unique(y_true)
  n_classes = len(unique_labels)
  confusion_matrix = np.zeros((n_classes, n_classes))
  for i in range(n_classes):
    for j in range(n_classes):
      confusion_matrix[i, j] = np.sum((y_true == unique_labels[i]) & (y_pred == unique_labels[j]))
  return confusion_matrix

def recall_score(y_true, y_pred, average='macro'):
  """
  Calcula o recall do modelo.

  Parâmetros:
    y_true: array-like, rótulos verdadeiros.
    y_pred: array-like, rótulos preditos.
    average: str, tipo de média a ser utilizada ('macro' ou 'micro').

  Retorna:
    Recall do modelo.
  """
  confusion_matrix = confusion_matrix(y_true, y_pred)
  recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
  if average == 'macro':
    return np.mean(recall)
  elif average == 'micro':
    return np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
  else:
    raise ValueError("Argumento 'average' inválido.")

def precision_score(y_true, y_pred, average='macro'):
  """
  Calcula a precisão do modelo.

  Parâmetros:
    y_true: array-like, rótulos verdadeiros.
    y_pred: array-like, rótulos preditos.
    average: str, tipo de média a ser utilizada ('macro' ou 'micro').

  Retorna:
    Precisão do modelo.
  """
  confusion_matrix = confusion_matrix(y_true, y_pred)
  precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
  if average == 'macro':
    return np.mean(precision)
  elif average == 'micro':
    return np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
  else:
    raise ValueError("Argumento 'average' inválido.")

def f1_score(y_true, y_pred, average='macro'):
  """
  Calcula o f1-score do modelo.

  Parâmetros:
    y_true: array-like, rótulos verdadeiros.
    y_pred: array-like, rótulos preditos.
    average: str, tipo de média a ser utilizada ('macro' ou 'micro').

  Retorna:
    F1-score do modelo.
  """
  precision = precision_score(y_true, y_pred, average=average)
  recall = recall_score(y_true, y_pred, average=average)
  return 2 * (precision * recall) / (precision + recall)