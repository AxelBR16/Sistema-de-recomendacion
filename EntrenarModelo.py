import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Cargar el archivo CSV
file_path = 'resultados_por_interes_normalizado.csv'  # Cambiar a resultados_por_aptitudes.csv si es necesario
data = pd.read_csv(file_path)

# Selección de características y variable objetivo
X = data[['Biologicos','Mecanicos','Campestres','Geofisicos','Sociales','Literarios','Organizacion','Persuasivo','Calculo','Contabilidad','Musical','Artistico','Cientificos']]  
y = data['Carrera']

# Dividir el conjunto de datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo de Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Realizar predicciones y evaluar precisión
y_pred = random_forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Precisión del modelo: {accuracy * 100:.2f}%")
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))

# Guardar el modelo entrenado
joblib.dump(random_forest, 'modelo_random_forest.pkl')
