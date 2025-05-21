from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Crear la aplicación FastAPI
app = FastAPI()

# Cargar el modelo entrenado con intereses
try:
    print("Cargando el modelo de intereses...")
    modelo = joblib.load('modelo_random_forest_intereses.pkl')  # Asegúrate de que este archivo exista
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

# Mapeo de las predicciones a las carreras
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

        # Convertir datos a DataFrame
        datos_entrada = pd.DataFrame([data.dict()])
        print("Datos de entrada convertidos a DataFrame:")
        print(datos_entrada)

        # Capitalizar para que coincidan con las columnas del modelo
        datos_entrada.columns = [col.capitalize() for col in datos_entrada.columns]
        print("Columnas después de capitalizar:")
        print(datos_entrada.columns)

        # Realizar la predicción
        prediccion = modelo.predict(datos_entrada)
        prediccion = int(prediccion[0])
        carrera_recomendada = carreras.get(prediccion, "Carrera no encontrada")

        return {
            "recomendacion_numerica": prediccion,
            "recomendacion": carrera_recomendada
        }

    except Exception as e:
        print(f"Error al hacer la predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al hacer la predicción: {str(e)}")

# Para ejecutar la API:
# uvicorn main:app --reload
