import pandas as pd

# Función para convertir los índices a cadenas de texto
def to_str(col_list):
    return [str(i) for i in col_list]

# Cargar el archivo CSV
df = pd.read_csv("C:/Users/leica/OneDrive/Documentos/Modelo/Sistema-de-recomendacion/Modelos actualizads/modelo de aptitudes/resultado_aptitudes.csv")

# Definir las categorías con sus índices
abstracta = [1,21,36,38,51,70,78,93,96,107]      
coordinacion = [2,13,35,44,56,65,74,88,101,114]       
numerica = [3,15,25,43,55,66,73,89,100,112]      
verbal = [4,16,27,41,53,68,80,91,98,110]
persuasiva = [5,20,34,46,50,71,77,85,104,117]
mecanica = [6,19,29,40,58,63,76,86,103,116]
social = [7,14,33,37,52,69,79,92,97,108]
directiva = [8,18,30,39,57,64,75,87,102,115]
organizacion = [9,17,26,42,54,67,81,90,99,111]
musical = [10,31,47,60,72,83,105,109,113,118]
artistico_plastico = [11,23,24,32,49,61,82,94,106,120]
espacial = [12,22,28,45,48,59,62,84,95,119]

# Convertir las listas de índices a cadenas usando la función to_str()
abstracta_cols = to_str(abstracta)
coordinacion_cols = to_str(coordinacion)
numerica_cols = to_str(numerica)
verbal_cols = to_str(verbal)
persuasiva_cols = to_str(persuasiva)
mecanica_cols = to_str(mecanica)
social_cols = to_str(social)
directiva_cols = to_str(directiva)
organizacion_cols = to_str(organizacion)
musical_cols = to_str(musical)
artistico_plastico_cols = to_str(artistico_plastico)
espacial_cols = to_str(espacial)

# Crear un DataFrame vacío para los resultados
df_resultado = pd.DataFrame()

# Sumar las columnas para cada categoría
df_resultado['abstracta'] = df[abstracta_cols].sum(axis=1)
df_resultado['coordinacion'] = df[coordinacion_cols].sum(axis=1)
df_resultado['numerica'] = df[numerica_cols].sum(axis=1)
df_resultado['verbal'] = df[verbal_cols].sum(axis=1)
df_resultado['persuasiva'] = df[persuasiva_cols].sum(axis=1)
df_resultado['mecanica'] = df[mecanica_cols].sum(axis=1)
df_resultado['social'] = df[social_cols].sum(axis=1)
df_resultado['directiva'] = df[directiva_cols].sum(axis=1)
df_resultado['organizacion'] = df[organizacion_cols].sum(axis=1)
df_resultado['musical'] = df[musical_cols].sum(axis=1)
df_resultado['artistico'] = df[artistico_plastico_cols].sum(axis=1)
df_resultado['espacial'] = df[espacial_cols].sum(axis=1)

# Si la columna 'Carrera' existe, añadirla al resultado
if 'Carrera' in df.columns:
    df_resultado.insert(0, 'Carrera', df['Carrera'])

# Guardar el resultado en un archivo CSV
df_resultado.to_csv("C:/Users/leica/OneDrive/Documentos/Modelo/Sistema-de-recomendacion/Modelos actualizads/modelo de aptitudes/resultados_procesado_aptitudes.csv", index=False)

print("Resultados guardados")
