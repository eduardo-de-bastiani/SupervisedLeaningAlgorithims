class DecisionTree:

  def __init__(self):
    self.root = None

  def fit(self, X, y):
    # Criar o nó raiz
    self.root = self._build_tree(X, y)

  def predict(self, X):
    y_pred = []
    for x in X:
      y_pred.append(self._predict_node(self.root, x))

    return y_pred

  def _build_tree(self, X, y):
    # Se todos os exemplos tiverem a mesma classe, retornar um nó folha
    if np.all(y == y[0]):
      return Node(label=y[0])

    # Se não, encontrar a melhor feature para dividir o espaço de features
    best_feature, best_threshold = self._find_best_split(X, y)

    # Criar um nó interno com a melhor feature como atributo de divisão
    node = Node(feature=best_feature, threshold=best_threshold)

    # Dividir o espaço de features em subconjuntos
    X_left, X_right, y_left, y_right = self._split_data(X, y, best_feature, best_threshold)

    # Criar os nós filhos esquerdo e direito
    node.left = self._build_tree(X_left, y_left)
    node.right = self._build_tree(X_right, y_right)

    return node

  def _predict_node(self, node, x):
    # Se for um nó folha, retornar a classe
    if node.is_leaf():
      return node.label

    # Se não, seguir para o nó filho esquerdo ou direito
    if x[node.feature] > node.threshold:
      return self._predict_node(node.right, x)
    else:
      return self._predict_node(node.left, x)

  def _find_best_split(self, X, y):
    # Calcular a entropia do conjunto de dados
    entropy = self._entropy(y)

    # Inicializar as variáveis para armazenar a melhor feature e threshold
    best_feature = None
    best_threshold = None

    # Percorrer todas as features
    for feature in range(X.shape[1]):
      # Encontrar o melhor threshold para a feature
      thresholds = np.unique(X[:, feature])
      for threshold in thresholds:
        # Calcular a entropia dos subconjuntos
        entropy_left, entropy_right = self._entropy_split(X, y, feature, threshold)

        # Calcular o ganho de informação
        information_gain = entropy - (entropy_left + entropy_right)

        # Atualizar a melhor feature e threshold se o ganho de informação for maior
        if information_gain > best_information_gain:
          best_information_gain = information_gain
          best_feature = feature
          best_threshold = threshold

    return best_feature, best_threshold

  def _split_data(self, X, y, feature, threshold):
    # Encontrar os índices dos exemplos que pertencem ao subconjunto esquerdo e direito
    indices_left = X[:, feature] <= threshold
    indices_right = X[:, feature] > threshold

    # Separar os exemplos em subconjuntos
    X_left = X[indices_left]
    X_right = X[indices_right]
    y_left = y[indices_left]
    y_right = y[indices_right]

    return X_left, X_right, y_left, y_right

  def _entropy(self, y):
    # Calcular a proporção de exemplos de cada classe
    p_classes = np.bincount(y) / len(y)

    # Calcular a entropia
    entropy = 0
    for p in p_classes:
      if p > 0:
        entropy += p * np.log2(p)

    return -entropy

  def _entropy_split(self, X, y, feature, threshold):
    # Encontrar os índices dos exemplos que pertencem ao subconjunto esquerdo e direito
    indices_left = X[:, feature] <= threshold
    indices_right = X[:, feature] > threshold

    # Separar os exemplos em subconjuntos
    y_left = y[indices_left]
    y_right = y[indices_right]