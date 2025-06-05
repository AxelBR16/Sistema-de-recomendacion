import pandas as pd

#Remplazar las respuestas por numeros segun la rubrica

archivo_csv = "C:/Users/leica/OneDrive/Documentos/Modelo/Sistema-de-recomendacion/Modelos actualizads/modelo de aptitudes/datos_aptitudes.csv"
df = pd.read_csv(archivo_csv) 

reemplazos = {
    'Ingeniería en sistemas computacionales': 0,
    'Licenciatura en Ciencia de Datos': 1,
    'Ingeniería en Inteligencia Artificial': 2,
    #preguntas
    'Mucho muy hábil': 5,
    'Muy hábil': 4,
    'Medianamente hábil': 3,
    'Poco hábil': 2,
    'Nada hábil': 0
}

df.replace(reemplazos, inplace=True)

df.to_csv("C:/Users/leica/OneDrive/Documentos/Modelo/Sistema-de-recomendacion/Modelos actualizads/modelo de aptitudes/resultado_aptitudes.csv")

print("Archivo procesado correctamente y guardado como 'resultado_aptitudes.csv'")