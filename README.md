Integrantes do Grupo:
~ Eduardo De Bastiani 
    matricula: 23106295
~ Carlos de Moraes
    matricula: 



Prompts utilizados:
Preciso que você crie algoritmos para três métodos de aprendizado:
- kNN
- Naïve Bayes
- Árvores de Decisão

Importante: os algoritmos implementados não devem usar bibliotecas com implementações prontas (como por exemplo scikit-learn). Quero uma implementação from scratch desses algoritmos, usando apenas bibliotecas auxiliares para computação vetorial (por exemplo, numpy).
Preciso da aplicação dos algoritmos criados em apenas 1 conjunto de dados e uma avaliação de seus resultados com matriz de confusão, acurácia, recall, precisão,  e f1-score. Vou sugerir alguns datasets para teste:  Iris, Penguins, Titanic, Census Income.
Também preciso da análise crítica do código gerado, sendo destacados:
- pontos fortes
- potenciais problemas
- diferenças entre as implementações 





###### não fará parte do arquivo readme
Lista de Tarefas e Progresso
Tarefa: Criar algoritmos para três métodos de aprendizado:

kNN:

Feito:
Implementação completa do algoritmo kNN com:
Função euclidean_distance para calcular a distância entre pontos.
Armazenamento dos dados de treinamento em self.X_train e self.y_train.
Cálculo das distâncias entre o novo ponto e os pontos de treinamento.
Busca pelos k vizinhos mais próximos.
Predição da classe através da votação majoritária entre os vizinhos.
Pendente:
Aplicação do kNN em um conjunto de dados.
Avaliação dos resultados com matriz de confusão, acurácia, recall, precisão e f1-score. ✅


# Naïve Bayes:

Feito:
Implementação completa do algoritmo Naïve Bayes com:
Cálculo das probabilidades a priori das classes.
Cálculo das médias e desvios padrões das features para cada classe.
Cálculo da probabilidade posterior para cada classe usando a distribuição normal.
Predição da classe com a maior probabilidade posterior.
Pendente:
Aplicação do Naïve Bayes em um conjunto de dados.
Avaliação dos resultados com matriz de confusão, acurácia, recall, precisão e f1-score. ✅


# Árvores de Decisão:

Feito:
Implementação completa da árvore de decisão com:
Funções para construir a árvore (_build_tree) e prever a classe (_predict_node).
Cálculo da entropia e do ganho de informação para encontrar a melhor feature para dividir o espaço de features.
Suporte para árvores binárias com um nó raiz, nós internos e nós folha.
Pendente:
Implementação da poda de árvores (opcional).
Aplicação da árvore de decisão em um conjunto de dados.
Avaliação dos resultados com matriz de confusão, acurácia, recall, precisão e f1-score. ✅
Tarefa: Aplicação dos algoritmos em um conjunto de dados.

Pendente:
Selecionar um conjunto de dados (Iris, Penguins, Titanic, Census Income).
Dividir o conjunto de dados em treino e teste.
Aplicar os algoritmos kNN, Naïve Bayes e Árvore de Decisão ao conjunto de dados de treino.
Fazer previsões para o conjunto de dados de teste.
Tarefa: Avaliação dos resultados com matriz de confusão, acurácia, recall, precisão e f1-score.

Pendente:
Calcular a matriz de confusão para cada algoritmo.
Calcular a acurácia, recall, precisão e f1-score para cada algoritmo.
Tarefa: Análise crítica do código gerado.

Pendente:
Descrever os pontos fortes de cada algoritmo.
Identificar os potenciais problemas de cada algoritmo.
Descrever as diferenças entre as implementações dos algoritmos.
Tarefa: Gerar o código em um arquivo Jupyter Notebook.

Pendente:
Implementar os algoritmos em um Jupyter Notebook.
Documentar o código de forma clara e organizada.
Observações:

O código está quase completo, apenas as partes de aplicação e avaliação ainda precisam ser implementadas.
O arquivo Jupyter Notebook ainda não foi criado.
A análise crítica do código será feita após a finalização da implementação e avaliação.
Próximos Passos:

Selecionar o conjunto de dados a ser utilizado.
Dividir o conjunto de dados em treino e teste.
Aplicar os algoritmos kNN, Naïve Bayes e Árvore de Decisão ao conjunto de dados de treino.
Fazer previsões para o conjunto de dados de teste.
Calcular a matriz de confusão, acurácia, recall, precisão e f1-score para cada algoritmo.
Descrever os pontos fortes, potenciais problemas e diferenças entre as implementações dos algoritmos.
Gerar o código em um Jupyter Notebook com documentação clara e organizada.
# SupervisedLeaningAlgorithims
