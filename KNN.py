import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score


file_path = 'resultados_por_interes.csv'
data = pd.read_csv(file_path)

print(data.head())

X = data[['Biologicos','Mecanicos','Campestres','Geofisicos',
          'Sociales','Literarios','Organizacion','Persuasivo',
          'Calculo','Contabilidad','Musical','Artistico','Cientificos']]

y = data['Carrera']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=5)  # Puedes cambiar el número de vecinos

knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo KNN: {accuracy * 100:.2f}%")

f1 = f1_score(y_test, y_pred, average='weighted')  # También puedes probar con 'macro' o 'micro'
print(f"F1-score del modelo KNN: {f1:.4f}")

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
