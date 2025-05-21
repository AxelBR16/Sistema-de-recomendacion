import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Cargar datos normalizados
data = pd.read_csv('resultados_por_aptitudes_normalizado.csv')

# Características y etiqueta
X = data[['abstracta', 'coordinacion', 'numerica', 'verbal', 'persuasiva', 'mecanica', 
          'social', 'directiva', 'organizacion', 'musical', 'artistico', 'espacial']]
y = data['Carrera']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir modelo base
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Definir grid de hiperparámetros para buscar
param_grid = {
    'n_estimators': [100, 200, 300],         # Número de árboles
    'max_depth': [None, 10, 20, 30],         # Profundidad máxima del árbol
    'min_samples_split': [2, 5, 10],         # Mínimo número de muestras para dividir nodo
    'min_samples_leaf': [1, 2, 4]             # Mínimo número de muestras en hoja
}

# Configurar GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Ejecutar búsqueda
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_rf = grid_search.best_estimator_

print("Mejores hiperparámetros encontrados:", grid_search.best_params_)

# Evaluar en el set de prueba
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Precisión con el mejor modelo: {accuracy * 100:.2f}%")
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))

# Guardar modelo optimizado
joblib.dump(best_rf, 'modelo_random_forest_aptitudes_optimizado.pkl')

print("Modelo optimizado guardado como 'modelo_random_forest_aptitudes_optimizado.pkl'")
