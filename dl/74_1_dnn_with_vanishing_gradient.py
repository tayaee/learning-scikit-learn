from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

input_dim = 100
output_dim = 1

# v1 model: Sigmoid activation function, initialize Glorot Uniform (default)
# Deeper layers increase the likelihood of slope vanishing
model_v1 = Sequential([
    Dense(64, activation="sigmoid", input_shape=(input_dim,)),
    Dense(64, activation="sigmoid"),
    Dense(64, activation="sigmoid"),
    Dense(64, activation="sigmoid"),  # A deep layer
    Dense(output_dim, activation="sigmoid"),
])

model_v1.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("--- DNN v1 (Sigmoid) ---")
model_v1.summary()
