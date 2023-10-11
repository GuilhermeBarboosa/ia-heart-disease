import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

# Carregue os dados
data = pd.read_csv("heart.csv")
X = data.drop('Heart Disease', axis=1)
y = data['Heart Disease']

# KNN 1
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
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

print("Acurácia do modelo KNN 1:", accuracy)
print("Tempo de execução KNN 1:", elapsed_time, "segundos \n")
# print(classification_report(y_test, y_pred))

# KNN 2
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
predictions = knn.predict(X)
accuracy = np.mean(predictions == y)
end_time = time.time()
elapsed_time = end_time - start_time
print("Acurácia do modelo KNN 2:", accuracy)
print("Tempo de execução KNN 2:", elapsed_time, "segundos \n")



# SVM 1
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
end_time = time.time()
elapsed_time = end_time - start_time

print("Acurácia do modelo SVM 1:", accuracy)
print("Tempo de execução SVM 1:", elapsed_time, "segundos \n")
# print(classification_report(y_test, y_pred))

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
print("Acurácia do modelo SVM 2:", accuracy_score(y_test, y_pred))
print("Tempo de execução SVM 2:", elapsed_time, "segundos \n")


# Redes Neurais Artificiais 1
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
end_time = time.time()
elapsed_time = end_time - start_time
print("Acurácia do modelo de RNA 1:", accuracy)
print("Tempo de execução RNA 1:", elapsed_time, "segundos \n")
# print(classification_report(y_test, y_pred))

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
print("Acurácia do modelo de RNA 2:", accuracy_score(y_test, y_pred))
print("Tempo de execução RNA 2:", elapsed_time, "segundos \n")


#Regressão logistica 1
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
end_time = time.time()
elapsed_time = end_time - start_time
print("Acurácia do modelo de Regressão Logística 1:", accuracy)
print("Tempo de execução Regressão Logística 1:", elapsed_time, "segundos \n")
# print(classification_report(y_test, y_pred))


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
print("Acurácia do modelo de Regressão Logística 2:", accuracy_score(y_test, y_pred))
print("Tempo de execução Regressão Logística 2:", elapsed_time, "segundos \n")