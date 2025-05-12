from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Crear la aplicación FastAPI
app = FastAPI()

# Cargar el modelo entrenado
try:
    print("Cargando el modelo...")
    modelo = joblib.load('modelo_random_forest.pkl')
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Error al cargar el modelo: {str(e)}")

# Definir la estructura de entrada para la API
class RecomendacionRequest(BaseModel):
    biologicos: float
    mecanicos: float
    campestres: float
    geofisicos: float
    sociales: float
    literarios: float
    organizacion: float
    persuasivo: float
    calculo: float
    contabilidad: float
    musical: float
    artistico: float
    cientificos: float

# Definir el mapeo de las predicciones a las carreras
carreras = {
    0: 'Ingeniería en sistemas computacionales',
    1: 'Licenciatura en Ciencia de Datos',
    2: 'Ingeniería en Inteligencia Artificial'
}

# Ruta para hacer predicciones
@app.post("/recomendacion/")
async def obtener_recomendacion(data: RecomendacionRequest):
    try:
        print("Recibiendo solicitud de recomendación...")
        # Convertir los datos de la solicitud en un DataFrame
        datos_entrada = pd.DataFrame([data.dict()])
        print("Datos de entrada convertidos a DataFrame:")
        print(datos_entrada)

        # Convertir todos los nombres de las columnas a mayúsculas para que coincidan con los nombres del entrenamiento
        datos_entrada.columns = [col.capitalize() for col in datos_entrada.columns]
        print("Columnas después de convertir a mayúsculas:")
        print(datos_entrada.columns)

        # Realizar la predicción
        prediccion = modelo.predict(datos_entrada)
        print(f"Predicción realizada: {prediccion}")

        # Convertir la predicción a un tipo JSON-serializable
        prediccion = int(prediccion[0])  # Convertir de numpy.int64 a int de Python
        print(f"Predicción convertida a tipo int: {prediccion}")

        # Mapeo de la predicción numérica a la carrera correspondiente
        carrera_recomendada = carreras.get(prediccion, "Carrera no encontrada")
        print(f"Carrera recomendada: {carrera_recomendada}")

        # Retornar tanto el valor de la predicción como el nombre de la carrera
        return {"recomendacion_numerica": prediccion, "recomendacion": carrera_recomendada}
    
    except Exception as e:
        print(f"Error al hacer la predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al hacer la predicción: {str(e)}")

# Para ejecutar la API, usa el siguiente comando en la terminal:
# uvicorn main:app --reload
