class KNN:

  def __init__(self, k):
    self.k = k

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y

  def predict(self, X):
    y_pred = []
    for x in X:
      # Encontrar os k pontos mais próximos
      distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
      k_nearest_indices = np.argsort(distances)[:self.k]
      k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

      # Classificar o novo ponto de dados
      majority_label = np.argmax(np.bincount(k_nearest_labels))
      y_pred.append(majority_label)

    return y_pred