import pandas as pd

df = pd.read_csv("archivo_modificado.csv")

biologicos = [2,14,35,45,57,74,90,92,105,128]      
mecanicos = [3,18,34,50,65,70,82,100,116,130]       
campestres = [4,15,38,46,56,75,88,93,109,124]      

biologicos_cols = [str(i) for i in biologicos]
mecanicos_cols = [str(i) for i in mecanicos]
campestres_cols = [str(i) for i in campestres]

df_resultado = pd.DataFrame()
df_resultado['Biológicos'] = df[biologicos_cols].sum(axis=1)
df_resultado['Mecánicos'] = df[mecanicos_cols].sum(axis=1)
df_resultado['Campestres'] = df[campestres_cols].sum(axis=1)

if 'Nombre' in df.columns:
    df_resultado.insert(0, 'Nombre', df['Nombre'])

# Guarda el resultado
df_resultado.to_csv("resultados_por_interes.csv", index=False)

print("Resultados guardados")
