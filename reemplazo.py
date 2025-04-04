import pandas as pd

#Remplazar emociones por numeros segun la rubrica

archivo_csv = "intereses.csv" 
df = pd.read_csv(archivo_csv) 

reemplazos = {
    'Me gusta mucho.': 5,
    'Me gusta.': 4,
    'Me es indiferente.': 3,
    'Me desagrada.': 2,
    'Me desagrada totalmente.': 1
}

df.replace(reemplazos, inplace=True)

df.to_csv("archivo_modificado.csv")

print("Archivo procesado correctamente y guardado como 'archivo_modificado.csv'")