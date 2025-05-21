import requests
import numpy as np

url = "http://127.0.0.1:8000/recomendacion/"

# Nombres de las características
features = ["abstracta", "coordinacion", "numerica", "verbal", "persuasiva", "mecanica",
            "social", "directiva", "organizacion", "musical", "artistico", "espacial"]

# Generar 10 ejemplos aleatorios con valores en [0,1] pero con rangos diferentes
# Ejemplo 1-3 valores bajos, 4-6 valores medios, 7-10 valores altos para diferenciar mejor
ejemplos = []
for i in range(10):
    if i < 3:
        # valores bajos
        ejemplo = {f: np.random.uniform(0.0, 0.3) for f in features}
    elif i < 6:
        # valores medios
        ejemplo = {f: np.random.uniform(0.3, 0.7) for f in features}
    else:
        # valores altos
        ejemplo = {f: np.random.uniform(0.7, 1.0) for f in features}
    ejemplos.append(ejemplo)

# Enviar las peticiones y mostrar resultados
for i, ejemplo in enumerate(ejemplos):
    response = requests.post(url, json=ejemplo)
    if response.status_code == 200:
        resultado = response.json()
        print(f"Ejemplo {i+1}: Predicción: {resultado['recomendacion_numerica']} - {resultado['recomendacion']}")
    else:
        print(f"Ejemplo {i+1}: Error en la petición. Código: {response.status_code}")
