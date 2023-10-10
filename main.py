import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn import svm

# Carregue os dados
data = pd.read_csv("heart.csv")

# KNN 1
X = data.drop('Heart Disease', axis=1)
y = data['Heart Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo KNN 1:", accuracy)
# print(classification_report(y_test, y_pred))

# KNN 2
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
predictions = knn.predict(X)
accuracy = np.mean(predictions == y)
print("Acurácia do modelo KNN 2:", accuracy)


# SVM 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo SVM 1:", accuracy)
# print(classification_report(y_test, y_pred))

# SVM 2
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
clf = svm.SVC(kernel="linear")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Acurácia do modelo SVM 2:", accuracy_score(y_test, y_pred))



# Redes Neurais Convolucionais (CNN)
# Redes Neurais Artificiais