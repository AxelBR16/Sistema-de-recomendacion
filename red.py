from tensorflow import Sequential
from tensorflow import Dense


import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Cargar datos
df = pd.read_csv('resultados_por_interes.csv')

# Separar entrada y salida
X = df.drop(columns=['Carrera'])
y = df['Carrera']

# Convertir etiquetas de texto a números
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(32, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(y_categorical.shape[1], activation='softmax'))  # Una neurona por clase

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)
loss, acc = model.evaluate(X_test, y_test)
print(f'Precisión: {acc*100:.2f}%')

# Para predecir una nueva muestra
nueva_muestra = X.iloc[0:1]  # por ejemplo, la primera fila
pred = model.predict(nueva_muestra)
pred_clase = le.inverse_transform([pred.argmax()])
print("Predicción:", pred_clase[0])
