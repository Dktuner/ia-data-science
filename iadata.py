import tensorflow as tf
import numpy as np

# Dados de entrada e saída para a porta lógica XOR
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

# Definição do modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilação do modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
model.fit(inputs, outputs, epochs=5000, verbose=2)

# Avaliação do modelo
loss, accuracy = model.evaluate(inputs, outputs)
print(f"Acurácia do modelo: {accuracy}")
