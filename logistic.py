import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

# Cargar los datos
file_path = 'resultados_por_interes_normalizado.csv'
data = pd.read_csv(file_path)

# Definir las características (X) y la variable objetivo (y)
X = data[['Biologicos','Mecanicos','Campestres','Geofisicos',
          'Sociales','Literarios','Organizacion','Persuasivo',
          'Calculo','Contabilidad','Musical','Artistico','Cientificos']]

y = data['Carrera']

# Escalar los datos (muy importante para redes neuronales)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Crear el modelo MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Predicciones
y_pred = mlp.predict(X_test)

# Evaluación
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precisión del modelo MLP: {accuracy * 100:.2f}%")
print(f"F1-score del modelo MLP: {f1:.4f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
