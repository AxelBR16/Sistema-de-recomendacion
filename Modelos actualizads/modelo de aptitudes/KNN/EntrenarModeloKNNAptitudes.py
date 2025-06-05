import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import joblib

# Cargar datos originales
df = pd.read_csv("C:/Users/leica/OneDrive/Documentos/Modelo/Sistema-de-recomendacion/Modelos actualizads/modelo de aptitudes/resultados_aptitudes_normalizado.csv")

X = df.drop(columns=['Carrera'])
y = df['Carrera']

# Normalizar datos
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# Guardar scaler para la API
joblib.dump(scaler,"C:/Users/leica/OneDrive/Documentos/Modelo/Sistema-de-recomendacion/Modelos actualizads/modelo de aptitudes/KNN/scaler_aptitudes.pkl")

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)

# Aplicar SMOTE solo al set de entrenamiento
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Distribución original en train:", pd.Series(y_train).value_counts())
print("Distribución tras SMOTE en train:", pd.Series(y_train_res).value_counts())

# Definir modelo base k-NN
knn = KNeighborsClassifier()

# Parámetros para GridSearch
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'euclidean', 'manhattan']
}

# GridSearchCV para buscar mejores hiperparámetros
grid_search = GridSearchCV(knn, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_res, y_train_res)

best_knn = grid_search.best_estimator_

print("Mejores hiperparámetros:", grid_search.best_params_)

# Evaluar en set de prueba original (sin SMOTE)
y_pred = best_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Precisión k-NN con SMOTE: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Guardar modelo optimizado
joblib.dump(best_knn,  "C:/Users/leica/OneDrive/Documentos/Modelo/Sistema-de-recomendacion/Modelos actualizads/modelo de aptitudes/KNN/modelo_knn_aptitudes_optimizado_smote.pkl")
print("Modelo k-NN optimizado y guardado con SMOTE.")
