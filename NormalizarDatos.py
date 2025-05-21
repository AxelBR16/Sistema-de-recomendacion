import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Cargar los datos desde el archivo CSV
df = pd.read_csv('resultados_por_aptitudes.csv')

# Separar la columna objetivo 'Carrera'
y = df['Carrera']
X = df.drop(columns=['Carrera'])

# Aplicar Min-Max Scaling solo a las columnas de aptitudes
scaler = MinMaxScaler()
X_normalizado = scaler.fit_transform(X)

# Se crea un DataFrame con los datos normalizados
df_normalizado = pd.DataFrame(X_normalizado, columns=X.columns)

# Redondear a dos decimales
df_normalizado = df_normalizado.round(2)

# Insertar la columna 'Carrera' al inicio
df_normalizado.insert(0, 'Carrera', y)

# Guardar el DataFrame normalizado en un nuevo archivo
df_normalizado.to_csv('resultados_por_aptitudes_normalizado.csv', index=False)
print("Datos normalizados guardados en 'resultados_por_aptitudes_normalizado.csv'")

# Guardar scaler
joblib.dump(scaler, 'scaler_aptitudes.pkl')
print("Scaler guardado como 'scaler_aptitudes.pkl'")
