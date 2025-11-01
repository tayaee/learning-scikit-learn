# tensorflow_example.py
import tensorflow as tf

print("--- TensorFlow 예제 ---")
print(f"TensorFlow 버전: {tf.__version__}")

# 1. 텐서(Tensor) 생성
# tf.constant는 불변 텐서를 생성합니다.
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
print(f"TensorFlow 텐서 a:\n{a}")

# 2. 텐서 연산
c = tf.matmul(a, b)
print(f"행렬 곱셈 결과 c:\n{c}")

# 3. 자동 미분 (Autograd)
# tf.GradientTape를 사용하여 미분 연산을 추적합니다.
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x * x

dy_dx = tape.gradient(y, x)
print(f"dy/dx (x=3) = {dy_dx}")  # 2 * 3 = 6.0

# 4. 간단한 모델 구축 및 학습
# tf.keras.Sequential을 사용해 모델을 순차적으로 구성합니다.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=4, input_shape=(2,)),  # 입력 층
    tf.keras.layers.Dense(units=2),  # 출력 층
])
print("\nTensorFlow 모델 구조:")
model.summary()
