import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Cargar los datos de aptitudes
file_path = 'resultados_por_aptitudes.csv'
data = pd.read_csv(file_path)

# Ver las primeras filas del dataset
print(data.head())

# Seleccionar las características (aptitudes)
X = data[['abstracta', 'coordinacion', 'numerica', 'verbal', 'persuasiva', 'mecanica', 
          'social', 'directiva', 'organizacion', 'musical', 'artistico', 'espacial']] 

# La variable objetivo es la carrera
y = data['Carrera']

# Dividir los datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el clasificador Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
random_forest.fit(X_train, y_train)

# Realizar predicciones sobre el conjunto de prueba
y_pred = random_forest.predict(X_test)

# Calcular la precisión y mostrar el reporte de clasificación
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
