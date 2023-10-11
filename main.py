import numpy as np
import pandas as pd
import time
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

def limpa_string(string):
  string = string.replace(" ", "")
  string = string.replace(":", "-")

  index = string.find(".")
  if index != -1:
      return string[:index]
  else:
      return string

graficos = []
nomeGraficos = []


data = pd.read_csv("heart.csv")
X = data.drop('Heart Disease', axis=1)
y = data['Heart Disease']

# Grafico de outliers
graficos.append(plt.figure(figsize=(10, 6)))
sns.boxplot(data=X)
plt.title('Gráfico de Outliers Dataset')
plt.xticks(rotation=90)
plt.show()
nomeGraficos.append("Matriz de Outliers")
# Obtenha o número de dados aproveitados
n_dados = len(data)
print("Quantidade de dados utilizados: " + str(n_dados))
# =================================================================================================

#Grafico de quantidade de pessoas doentes
heart_disease_counts = data['Heart Disease'].value_counts()
graficos.append(plt.figure(figsize=(8, 6)))
bars = plt.bar(heart_disease_counts.index, heart_disease_counts.values, color=['lightblue', 'lightcoral'])
plt.xticks([0, 1], ['Sem Problemas Cardíacos', 'Com Problemas Cardíacos'])
plt.xlabel('Condição Cardíaca')
plt.ylabel('Número de Pessoas')
plt.title('Distribuição de Pessoas com e sem Problemas Cardíacos')

# Adicione os valores na legenda
for bar, count in zip(bars, heart_disease_counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2, count, str(count), ha='center', va='bottom')

plt.show()
nomeGraficos.append("Distribuição de Pessoas com e sem Problemas Cardíacos")
# =================================================================================================


# KNN 1
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
end_time = time.time()
elapsed_time = end_time - start_time
print('Tempo de execução do modelo KNN 1: {:.2f} segundos'.format(elapsed_time))
print('Acurácia do modelo KNN 1: {:.2f}%'.format(accuracy * 100))
print("\n")
# print(classification_report(y_test, y_pred))
graficos.append(plt.figure(figsize=(6, 4)))
confusion = confusion_matrix(y_test, y_pred)
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Valores Preditos')
plt.ylabel('Valores Reais')
plt.title('Matriz de Confusão do KNN 1')
plt.show()
nomeGraficos.append("Matriz de Confusão do KNN 1")
# =================================================================================================

# KNN 2
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
predictions = knn.predict(X)
accuracy = np.mean(predictions == y)
end_time = time.time()
elapsed_time = end_time - start_time
print('Tempo de execução do modelo KNN 2: {:.2f} segundos'.format(elapsed_time))
print('Acurácia do modelo KNN 2: {:.2f}%'.format(accuracy * 100))
print("\n")
confusion = confusion_matrix(y, predictions)
graficos.append(plt.figure(figsize=(6, 4)))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Valores Preditos')
plt.ylabel('Valores Reais')
plt.title('Matriz de Confusão do KNN 2')
plt.show()
nomeGraficos.append("Matriz de Confusão do KNN 2")
# =================================================================================================


# SVM 1
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
end_time = time.time()
elapsed_time = end_time - start_time
print('Tempo de execução do modelo SVM 1: {:.2f} segundos'.format(elapsed_time))
print('Acurácia do modelo SVM 1: {:.2f}%'.format(accuracy * 100))
print("\n")
# print(classification_report(y_test, y_pred))
confusion = confusion_matrix(y_test, y_pred)
graficos.append(plt.figure(figsize=(6, 4)))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Valores Preditos')
plt.ylabel('Valores Reais')
plt.title('Matriz de Confusão do SVM 1')
plt.show()
nomeGraficos.append("Matriz de Confusão do SVM 1")
# =================================================================================================

# SVM 2
start_time = time.time()
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
clf = svm.SVC(kernel="linear")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
end_time = time.time()
elapsed_time = end_time - start_time
print('Tempo de execução do modelo SVM 2: {:.2f} segundos'.format(elapsed_time))
print('Acurácia do modelo SVM 2: {:.2f}%'.format(accuracy * 100))
print("\n")
confusion = confusion_matrix(y_test, y_pred)
graficos.append(plt.figure(figsize=(6, 4)))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Valores Preditos')
plt.ylabel('Valores Reais')
plt.title('Matriz de Confusão do SVM 2')
plt.show()
nomeGraficos.append("Matriz de Confusão do SVM 2")
# =================================================================================================


# Redes Neurais Artificiais 1
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
end_time = time.time()
elapsed_time = end_time - start_time
print('Tempo de execução do modelo RNA 1: {:.2f} segundos'.format(elapsed_time))
print('Acurácia do modelo RNA 1: {:.2f}%'.format(accuracy * 100))
print("\n")
# print(classification_report(y_test, y_pred))
confusion = confusion_matrix(y_test, y_pred)
graficos.append(plt.figure(figsize=(6, 4)))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Valores Preditos')
plt.ylabel('Valores Reais')
plt.title('Matriz de Confusão do RNA 1')
plt.show()
nomeGraficos.append("Matriz de Confusão do RNA 1")
# =================================================================================================


# Redes Neurais Artificiais 2
start_time = time.time()
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model = MLPClassifier(hidden_layer_sizes=(1024, 512), activation='relu', solver='adam', max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
end_time = time.time()
elapsed_time = end_time - start_time
print('Tempo de execução do modelo RNA 2: {:.2f} segundos'.format(elapsed_time))
print('Acurácia do modelo RNA 2: {:.2f}%'.format(accuracy * 100))
print("\n")
confusion = confusion_matrix(y_test, y_pred)
graficos.append(plt.figure(figsize=(6, 4)))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Valores Preditos')
plt.ylabel('Valores Reais')
plt.title('Matriz de Confusão do RNA 2')
plt.show()
nomeGraficos.append("Matriz de Confusão do RNA 2")
# =================================================================================================



#Regressão logistica 1
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
end_time = time.time()
elapsed_time = end_time - start_time
print('Tempo de execução do modelo Regressão logistica 1: {:.2f} segundos'.format(elapsed_time))
print('Acurácia do modelo Regressão logistica 1: {:.2f}%'.format(accuracy * 100))
print("\n")
# print(classification_report(y_test, y_pred))
confusion = confusion_matrix(y_test, y_pred)
graficos.append(plt.figure(figsize=(6, 4)))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Valores Preditos')
plt.ylabel('Valores Reais')
plt.title('Matriz de Confusão da Regressão Logística 1')
plt.show()
nomeGraficos.append("Matriz de Confusão do Regressão Logística 1")
# =================================================================================================


#Regressão logistica 2
start_time = time.time()
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
end_time = time.time()
elapsed_time = end_time - start_time
print('Tempo de execução do modelo Regressão logistica 2: {:.2f} segundos'.format(elapsed_time))
print('Acurácia do modelo Regressão logistica 2: {:.2f}%'.format(accuracy * 100))
print("\n")
confusion = confusion_matrix(y_test, y_pred)
graficos.append(plt.figure(figsize=(6, 4)))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Valores Preditos')
plt.ylabel('Valores Reais')
plt.title('Matriz de Confusão da Regressão Logística 1')
plt.show()
nomeGraficos.append("Matriz de Confusão do Regressão Logística 2")
# =================================================================================================

for grafico in graficos:
    # hora_atual = limpa_string(str(datetime.datetime.now()))

    # titulo_da_janela = nomeGraficos[graficos.index(grafico)] + "-" + str(hora_atual)

    titulo_da_janela = nomeGraficos[graficos.index(grafico)]
    # Obtém o nome do arquivo
    nome_arquivo = "grafico_" + titulo_da_janela + ".png"

    # Salva o gráfico
    grafico.savefig(f"graficos/{nome_arquivo}")


