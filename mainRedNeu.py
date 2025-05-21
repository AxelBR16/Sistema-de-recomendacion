from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

app = FastAPI()

try:
    print("Cargando modelo red neuronal y scaler...")
    modelo = tf.keras.models.load_model('modelo_red_neuronal_aptitudes.h5')
    scaler = joblib.load('scaler_aptitudes.pkl')
    print("Modelo y scaler cargados correctamente.")
except Exception as e:
    print(f"Error al cargar modelo o scaler: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Error al cargar modelo o scaler: {str(e)}")

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
        datos_normalizados = scaler.transform(datos_entrada)
        pred_prob = modelo.predict(datos_normalizados)
        prediccion_int = int(np.argmax(pred_prob, axis=1)[0])
        carrera_recomendada = carreras.get(prediccion_int, "Carrera no encontrada")
        return {
            "recomendacion_numerica": prediccion_int,
            "recomendacion": carrera_recomendada
        }
    except Exception as e:
        print(f"Error al hacer la predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al hacer la predicción: {str(e)}")
