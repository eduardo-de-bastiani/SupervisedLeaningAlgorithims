class NaiveBayes:

  def __init__(self):
    self.class_priors = {}
    self.feature_means = {}
    self.feature_stddevs = {}

  def fit(self, X, y):
    # Calcular as probabilidades a priori das classes
    for label in np.unique(y):
      self.class_priors[label] = np.mean(y == label)

    # Calcular as médias e os desvios padrões das features para cada classe
    for label in np.unique(y):
      self.feature_means[label] = {}
      self.feature_stddevs[label] = {}
      for i in range(X.shape[1]):
        self.feature_means[label][i] = np.mean(X[y == label, i])
        self.feature_stddevs[label][i] = np.std(X[y == label, i])

  def predict(self, X):
    y_pred = []
    for x in X:
      # Calcular a probabilidade posterior para cada classe
      class_posteriors = {}
      for label in self.class_priors.keys():
        class_posteriors[label] = self.class_priors[label]
        for i in range(X.shape[1]):
          class_posteriors[label] *= normal_probability(x[i], self.feature_means[label][i], self.feature_stddevs[label][i])

      # Classificar o novo ponto de dados
      y_pred.append(max(class_posteriors, key=class_posteriors.get))

    return y_pred