import pandas as pd

# Ruta del archivo CSV de entrada (puedes cambiarla)
archivo_entrada = "resultados_por_interes.csv"

# Ruta del archivo CSV de salida
archivo_salida = "resultados_por_interes_normalizado.csv"

# Cargar los datos
df = pd.read_csv(archivo_entrada)

# Normalizar todos los valores numéricos al rango [0, 1]
df_normalizado = df.copy()

idx = df.columns.get_loc('Carrera')
X = df.iloc[:, idx+1:]

for columna in X.columns:
    # Intentar convertir a número, si falla, no se normaliza
    try:
        df_normalizado[columna] = pd.to_numeric(df[columna], errors='coerce') / 50
    except:
        pass

# Guardar en nuevo archivo CSV
df_normalizado.to_csv(archivo_salida, index=False)

print(f"Archivo normalizado guardado como: {archivo_salida}")
