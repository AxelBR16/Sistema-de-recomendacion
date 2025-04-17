import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

file_path = 'resultados_por_interes_normalizado.csv'
data = pd.read_csv(file_path)

print(data.head())

X = data[['Biologicos','Mecanicos','Campestres','Geofisicos','Sociales','Literarios','Organizacion','Persuasivo','Calculo','Contabilidad','Musical','Artistico','Cientificos']]  


y = data['Carrera'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

