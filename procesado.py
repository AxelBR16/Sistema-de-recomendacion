import pandas as pd

df = pd.read_csv("reemplazo.csv")
df.columns = df.columns.astype(str)  

biologicos = [2,14,35,45,57,74,90,92,105,128]      
mecanicos = [3,18,34,50,65,70,82,100,116,130]       
campestres = [4,15,38,46,56,75,88,93,109,124]   
geofisicos = [5,17,36,47,61,66,89,97,110,118]
SSocial = [6,19,28,41,63,68,84,98,111,123]
literarios = [7,21,33,44,58,76,87,94,106,127]
organizacion = [8,22,29,40,60,78,79,103,113,121]
persuasivo = [1,16,37,49,55,73,91,104,112,122]
calculo = [13,20,27,42,59,77,86,95,107,126]
contabilidad = [12,26,39,48,64,69,83,99,117,129]
musical = [9,23,30,52,54,71,81,101,115,119]
artistico_plastico = [11,25,32,51,53,72,80,102,114,120]
cientificos = [10,24,31,43,62,67,85,96,108,125]

def to_str(col_list):
    return [str(i) for i in col_list]

df_resultado = pd.DataFrame()
df_resultado['Biologicos'] = df[to_str(biologicos)].sum(axis=1)
df_resultado['Mecanicos'] = df[to_str(mecanicos)].sum(axis=1)
df_resultado['Campestres'] = df[to_str(campestres)].sum(axis=1)
df_resultado['Geofisicos'] = df[to_str(geofisicos)].sum(axis=1)
df_resultado['Sociales'] = df[to_str(SSocial)].sum(axis=1)
df_resultado['Literarios'] = df[to_str(literarios)].sum(axis=1)
df_resultado['Organizacion'] = df[to_str(organizacion)].sum(axis=1)
df_resultado['Persuasivo'] = df[to_str(persuasivo)].sum(axis=1)
df_resultado['Calculo'] = df[to_str(calculo)].sum(axis=1)
df_resultado['Contabilidad'] = df[to_str(contabilidad)].sum(axis=1)
df_resultado['Musical'] = df[to_str(musical)].sum(axis=1)
df_resultado['Artistico'] = df[to_str(artistico_plastico)].sum(axis=1)
df_resultado['Cientificos'] = df[to_str(cientificos)].sum(axis=1)

if 'Carrera' in df.columns:
    df_resultado.insert(0, 'Carrera', df['Carrera'])

df_resultado.to_csv("resultados_por_interes.csv", index=False)

print("Resultados guardados")
