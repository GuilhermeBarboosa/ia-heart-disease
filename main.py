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
tempos = []
graficoModelos = []
accuracyArray = []

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
percent_75 = int(0.75 * n_dados)
percent_25 = n_dados - percent_75

print("Quantidade de dados utilizados: " + str(n_dados))
print("75% dos dados (treinamento): " + str(percent_75))
print("25% dos dados (testes): " + str(percent_25))
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
graficoModelos.append("KNN 1")
tempos.append(elapsed_time)
accuracyArray.append(accuracy)
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
graficoModelos.append("KNN 2")
tempos.append(elapsed_time)
accuracyArray.append(accuracy)
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
confusion = confusion_matrix(y_test, y_pred)
graficos.append(plt.figure(figsize=(6, 4)))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Valores Preditos')
plt.ylabel('Valores Reais')
plt.title('Matriz de Confusão do SVM 1')
plt.show()
nomeGraficos.append("Matriz de Confusão do SVM 1")
graficoModelos.append("SVM 1")
tempos.append(elapsed_time)
accuracyArray.append(accuracy)
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
accuracy = accuracy_score(y_test, y_pred)
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
graficoModelos.append("SVM 2")
tempos.append(elapsed_time)
accuracyArray.append(accuracy)
# =================================================================================================


# Redes Neurais Artificiais 1
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', max_iter=1000, random_state=123)
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
graficoModelos.append("RNA 1")
tempos.append(elapsed_time)
accuracyArray.append(accuracy)
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
accuracy = accuracy_score(y_test, y_pred)
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
graficoModelos.append("RNA 2")
tempos.append(elapsed_time)
accuracyArray.append(accuracy)
# =================================================================================================



#Regressão logistica 1
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=456)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(random_state=456)
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
graficoModelos.append("Regressão Logística 1")
tempos.append(elapsed_time)
accuracyArray.append(accuracy)
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
accuracy = accuracy_score(y_test, y_pred)
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
graficoModelos.append("Regressão Logística 2")
tempos.append(elapsed_time)
accuracyArray.append(accuracy)
# =================================================================================================

# Grafico de tempo de execução dos modelos
graficos.append(plt.figure(figsize=(10, 6)))
plt.barh(graficoModelos, tempos, color='lightblue')
plt.xlabel('Tempo de Execução (segundos)')
plt.title('Tempo de Execução dos Modelos')
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Adicione os tempos como rótulos nas barras
for i, tempo in enumerate(tempos):
    plt.text(tempo, i, f'{tempo:.2f} s', va='center', fontsize=10, color='dimgrey')

plt.tight_layout()
plt.show()
nomeGraficos.append("Gráfico de Tempo de Execução dos Modelos")
# =================================================================================================

# Grafico de acuracia dos modelos
graficos.append(plt.figure(figsize=(10, 6)))
bars = plt.barh(graficoModelos, accuracyArray, color='lightblue')
plt.xlabel('Acurácia (%)')
plt.title('Acurácia dos Modelos')
for bar, accuracy in zip(bars, accuracyArray):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{accuracy:.2f}%', ha='left', va='center')
plt.show()
nomeGraficos.append("Gráfico de Acuracia dos Modelos")
# =================================================================================================
# Tabela de acuracia e tempo de execução dos modelos

formatted_accuracy = ["{:.3f}%".format(acc) for acc in accuracyArray]
formatted_tempos = ["{:.3f} seg".format(tp) for tp in tempos]

data = {
    "Modelo": graficoModelos,
    "Acurácia": formatted_accuracy,
    "Tempo (segundos)": formatted_tempos
}

df = pd.DataFrame(data)
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
plt.show()
graficos.append(fig)
nomeGraficos.append("Tabela dos modelos")

# =================================================================================================



for grafico in graficos:
    titulo_da_janela = nomeGraficos[graficos.index(grafico)]
    nome_arquivo = "grafico_" + titulo_da_janela + ".png"
    grafico.savefig(f"graficos/{nome_arquivo}")

# print(graficoModelos)
# print(accuracyArray)
# print(tempos)
data = {
    "Modelo": graficoModelos,
    "Acurácia": formatted_accuracy,
    "Tempo (segundos)": formatted_tempos
}

df = pd.DataFrame(data)

print(df)
