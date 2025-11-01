import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

MODEL_PATH = "my_binary_classification_model.keras"
print(f"Path to model: {MODEL_PATH}")
print("-" * 40)


X_train = np.random.rand(1000, 10).astype("float32")
y_train = np.random.randint(0, 2, size=(1000, 1)).astype("float32")

X_new = np.random.rand(5, 10).astype("float32")

model = keras.Sequential()
model.add(layers.Dense(32, activation="relu", input_shape=(10,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

sgd_optimizer = keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=sgd_optimizer, loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
print("-" * 40)
model.summary()

model.save(MODEL_PATH)
print(f"Saved model at {MODEL_PATH}")
print("-" * 40)


del model

loaded_model = keras.models.load_model(MODEL_PATH)

print(f"Loaded model from {MODEL_PATH}")
loaded_model.summary()
print("-" * 40)


print("Using the model...")
predictions_prob = loaded_model.predict(X_new, verbose=0)
predicted_classes = (predictions_prob > 0.5).astype(int)

print("\n[Prediction Probabilities]")
print(predictions_prob.flatten())

print("\n[Predicted Classes]")
print(predicted_classes.flatten())
print("-" * 40)

os.remove(MODEL_PATH)
print(f"Deleted file {MODEL_PATH}")
