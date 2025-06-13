import pickle
import numpy as np

# Cargar el modelo
with open('modelo_random_forest_aptitudes_optimizado.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Cargar el escalador
with open('./scaler_intereses.pkl', 'rb') as f:
    escalador = pickle.load(f)

# Datos de prueba (ejemplo)
# Supongamos que tienes un array de 2 características
# Reemplaza estos datos con los datos reales que quieres probar
datos_prueba = np.array([28, 34, 36, 30, 34, 36, 25, 33, 30, 38, 34, 34]).reshape(-1, 1)

# Escalar los datos de prueba usando el escalador cargado
datos_prueba_escalados = escalador.transform(datos_prueba)

# Realizar las predicciones con el modelo
predicciones = modelo.predict(datos_prueba_escalados)

print("Predicciones: ", predicciones)
