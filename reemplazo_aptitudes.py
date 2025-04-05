import pandas as pd

#Remplazar emociones por numeros segun la rubrica

archivo_csv = "aptitudes.csv" 
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
    'Nada hábil': 1
}

df.replace(reemplazos, inplace=True)

df.to_csv("resultado_aptitudes.csv")

print("Archivo procesado correctamente y guardado como 'archivo_modificado.csv'")