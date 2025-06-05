from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Cargar el modelo y el scaler guardados
try:
    print("Cargando el modelo y el scaler...")
    modelo = joblib.load("C:/Users/leica/OneDrive/Documentos/Modelo/Sistema-de-recomendacion/Modelos actualizads/modelo de aptitudes/RamdomForest/modelo_random_forest_aptitudes_optimizado.pkl")
    scaler = joblib.load("C:/Users/leica/OneDrive/Documentos/Modelo/Sistema-de-recomendacion/Modelos actualizads/modelo de aptitudes/scaler_aptitudes.pkl")  # scaler que normalizó los datos de entrenamiento
    print("Modelo y scaler cargados correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo o scaler: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Error al cargar el modelo o scaler: {str(e)}")

class RecomendacionRequest(BaseModel):
    abstracta: float
    coordinacion: float
    numerica: float
    verbal: float
    persuasiva: float
    mecanica: float
    social: float
    directiva: float
    organizacion: float
    musical: float
    artistico: float
    espacial: float

carreras = {
    0: 'Ingeniería en sistemas computacionales',
    1: 'Licenciatura en Ciencia de Datos',
    2: 'Ingeniería en Inteligencia Artificial'
}

@app.post("/recomendacion/")
async def obtener_recomendacion(data: RecomendacionRequest):
    try:
        datos_entrada = pd.DataFrame([data.dict()])

        # Normalizar con el scaler cargado
        datos_normalizados = scaler.transform(datos_entrada)

        # Crear DataFrame normalizado (con columnas originales)
        datos_normalizados_df = pd.DataFrame(datos_normalizados, columns=datos_entrada.columns)

        prediccion = modelo.predict(datos_normalizados_df)
        prediccion_int = int(prediccion[0])
        carrera_recomendada = carreras.get(prediccion_int, "Carrera no encontrada")

        return {
            "recomendacion_numerica": prediccion_int,
            "recomendacion": carrera_recomendada
        }

    except Exception as e:
        print(f"Error al hacer la predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al hacer la predicción: {str(e)}")
