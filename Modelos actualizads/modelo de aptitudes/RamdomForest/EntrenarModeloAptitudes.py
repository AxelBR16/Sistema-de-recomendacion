import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  # Para balanceo de clases
import joblib
from sklearn.preprocessing import StandardScaler  # Cambiado a StandardScaler

# Cargar datos normalizados
data = pd.read_csv("C:/Users/leica/OneDrive/Documentos/Modelo/Sistema-de-recomendacion/Modelos actualizads/modelo de aptitudes/resultados_aptitudes_normalizado.csv")

# Características y etiqueta
X = data[['abstracta', 'coordinacion', 'numerica', 'verbal', 'persuasiva', 'mecanica', 
          'social', 'directiva', 'organizacion', 'musical', 'artistico', 'espacial']]
y = data['Carrera']

# Normalizar datos con StandardScaler (cambio en el escalado)
scaler = StandardScaler()  # Usar StandardScaler en lugar de MinMaxScaler
X_norm = scaler.fit_transform(X)  # Ajustar el scaler y transformar los datos

# Guardar el scaler para la API
joblib.dump(scaler, "C:/Users/leica/OneDrive/Documentos/Modelo/Sistema-de-recomendacion/Modelos actualizads/modelo de aptitudes/RamdomForest/scaler_aptitudes.pkl")

# Balanceo de clases con SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_norm, y)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Definir modelo base
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Definir un grid de hiperparámetros más amplio
param_grid = {
    'n_estimators': [200, 300, 400, 500, 600],        # Más valores para n_estimators
    'max_depth': [None, 10, 20, 30, 40, 50, 60],       # Probar más profundidades
    'min_samples_split': [2, 5, 10, 15, 20],           # Probar más valores
    'min_samples_leaf': [1, 2, 4, 6, 8],               # Ajustar tamaño de hojas
    'max_features': ['auto', 'sqrt', 'log2'],          # Más opciones para la selección de características
    'bootstrap': [True, False]                          # Usar o no bootstrap
}

# Configurar GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Ejecutar búsqueda
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_rf = grid_search.best_estimator_

# Imprimir los mejores hiperparámetros encontrados
print("Mejores hiperparámetros encontrados:", grid_search.best_params_)

# Evaluar en el set de prueba
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Mostrar precisión y reporte de clasificación
print(f"Precisión con el mejor modelo: {accuracy * 100:.2f}%")
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))

# Guardar modelo optimizado
joblib.dump(best_rf, "C:/Users/leica/OneDrive/Documentos/Modelo/Sistema-de-recomendacion/Modelos actualizads/modelo de aptitudes/RamdomForest/modelo_random_forest_aptitudes_optimizado.pkl")

print("Modelo optimizado guardado como 'modelo_random_forest_aptitudes_optimizado.pkl'")
