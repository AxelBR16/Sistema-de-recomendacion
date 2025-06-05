import pandas as pd

#Remplazar emociones por numeros segun la rubrica

archivo_csv = "C:/Users/leica/OneDrive/Documentos/Modelo/Sistema-de-recomendacion/Modelos actualizads/modelo de Intereses/intereses.csv"
df = pd.read_csv(archivo_csv) 

reemplazos = {
    'Ingeniería en sistemas computacionales': 0,
    'Licenciatura en Ciencia de Datos': 1,
    'Ingeniería en Inteligencia Artificial': 2,
    'Me gusta mucho.': 5,
    'Me gusta.': 4,
    'Me es indiferente.': 3,
    'Me desagrada.': 2,
    'Me desagrada totalmente.': 1
}

df.replace(reemplazos, inplace=True)

df.to_csv("C:/Users/leica/OneDrive/Documentos/Modelo/Sistema-de-recomendacion/Modelos actualizads/modelo de Intereses/resultado_intereses.csv")

print("Archivo procesado correctamente y guardado como 'resultado_intereses.csv'")