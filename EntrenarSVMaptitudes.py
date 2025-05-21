import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.preprocessing import MinMaxScaler

# Cargar datos originales
df = pd.read_csv('resultados_por_aptitudes_normalizado.csv')

X = df.drop(columns=['Carrera'])
y = df['Carrera']

# Normalizar datos
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# Guardar scaler para la API
joblib.dump(scaler, 'scaler_aptitudes.pkl')

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)

# Aplicar SMOTE solo al set de entrenamiento
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Distribución original en train:", pd.Series(y_train).value_counts())
print("Distribución tras SMOTE en train:", pd.Series(y_train_res).value_counts())

# Definir modelo base SVM
svm = SVC(probability=True, random_state=42)

# Parámetros para GridSearch
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# GridSearchCV para buscar mejores hiperparámetros
grid_search = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_res, y_train_res)

best_svm = grid_search.best_estimator_

print("Mejores hiperparámetros:", grid_search.best_params_)

# Evaluar en set de prueba original (sin SMOTE)
y_pred = best_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Precisión SVM con SMOTE: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Guardar modelo optimizado
joblib.dump(best_svm, 'modelo_svm_aptitudes_optimizado_smote.pkl')
print("Modelo SVM optimizado y guardado con SMOTE.")
