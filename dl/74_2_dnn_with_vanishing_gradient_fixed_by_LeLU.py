from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

input_dim = 100
output_dim = 1

# v2 Model: ReLU Enable function, apply He Normal initialization
model_v2 = Sequential([
    Dense(64, activation="relu", input_shape=(input_dim,), kernel_initializer="he_normal"),  # apply He Normal reset
    Dense(64, activation="relu", kernel_initializer="he_normal"),
    Dense(64, activation="relu", kernel_initializer="he_normal"),
    Dense(64, activation="relu", kernel_initializer="he_normal"),
    # Output layer keeps appropriate activation function depending on the problem (e.g. Sigmoid for binary classification plane)
    Dense(output_dim, activation="sigmoid"),
])

model_v2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("\n--- DNN v2 (ReLU & He Normal) ---")
model_v2.summary()
