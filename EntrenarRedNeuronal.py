import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import numpy as np

# Cargar datos
df = pd.read_csv('resultados_por_aptitudes.csv')

X = df.drop(columns=['Carrera'])
y = df['Carrera']

# Normalizar
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler_aptitudes.pkl')

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)

# Aplicar SMOTE para balancear
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Convertir etiquetas a one-hot (si usas clasificación categórica)
num_classes = len(np.unique(y))
y_train_cat = tf.keras.utils.to_categorical(y_train_res, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

# Crear modelo simple
model = Sequential([
    Dense(64, input_dim=X_train_res.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping para evitar sobreentrenamiento
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar modelo
history = model.fit(X_train_res, y_train_cat, epochs=100, batch_size=16,
                    validation_split=0.2, callbacks=[early_stop], verbose=2)

# Evaluar modelo
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Precisión Red Neuronal: {accuracy * 100:.2f}%")

# Predicciones para reporte
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print(classification_report(y_test, y_pred))

# Guardar modelo y scaler
model.save('modelo_red_neuronal_aptitudes.h5')
print("Modelo guardado como 'modelo_red_neuronal_aptitudes.h5'")
