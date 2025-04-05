
import csv

# Datos proporcionados
data = [0,3,4,3,4,5,4,4,3,1,4,4,4,4,5,5,3,4,4,3,3,4,4,3,4,5,3,3,3,4,4,3,5,3,4,4,3,3,4,3,4,4,3,3,1,1,3,5,5,5,3,5,3,5,4,4,2,5,4,3,4,4,4,3,3,3,5,3,3,4,4,3,4,2,4,4,1,1,3,4,5,4,5,5,3,3,3,3,4,4,3,5,3,4,2,3,3,4,3,3,3,4,5,4,2,4,3,2,3,4,4,4,5,4,5,3,5,3,4,5,3,3,4,4,4,2,2,3,4,5,5]

# Crear el archivo CSV
with open('reemplazo.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    
    # Escribir los datos en una fila
    writer.writerow(data)

print("Archivo CSV creado exitosamente como 'resultados_test.csv'")