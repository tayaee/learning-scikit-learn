import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 1. Load Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Data Preprocessing (Normalization and Reshaping)
# --- Image Data Preprocessing ---
# Normalize pixel values to be between 0 and 1 (from 0-255)
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Reshape the data for CNN input: (samples, 28, 28) -> (samples, 28, 28, 1)
# 1 indicates the single channel for grayscale images
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# --- Label Data Preprocessing ---
# Convert labels (targets) to One-Hot Encoding
# Example: 5 -> [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


# 3. Build the CNN Model
model = Sequential([
    # First Convolutional Layer: 32 filters (3x3), 'relu' activation
    Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
    # Max Pooling Layer: Reduces size by taking the maximum value in 2x2 blocks
    MaxPooling2D(pool_size=(2, 2)),
    # Second Convolutional Layer
    Conv2D(64, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    # Flatten the 2D feature map into a 1D vector
    Flatten(),
    # Fully Connected Layer (Output Layer)
    # Uses 'softmax' to output probabilities for the 10 classes (0-9)
    Dense(num_classes, activation="softmax"),
])

# 4. Compile the Model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 5. Train the Model
batch_size = 128  # Number of samples processed per gradient update
epochs = 10  # Number of times the entire training dataset is iterated over
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# 6. Evaluate the Model
score = model.evaluate(x_test, y_test, verbose=0)
print("----------------------------------")
print(f"Test Loss: {score[0]:.4f}")
print(f"Test Accuracy: {score[1]:.4f}")
