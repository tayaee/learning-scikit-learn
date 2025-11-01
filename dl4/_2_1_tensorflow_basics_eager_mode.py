import tensorflow as tf
import numpy as np

# 1. 텐서 생성 (Tensor Creation)
# 텐서는 GPU 연산을 위한 다차원 배열입니다.
print("--- 텐서 생성 ---")
tensor_from_list = tf.constant([1, 2, 3, 4])
print(f"리스트로부터 생성: {tensor_from_list}")
print(f"데이터 타입: {tensor_from_list.dtype}")

tensor_from_numpy = tf.convert_to_tensor(np.array([[10, 20], [30, 40]]))
print(f"NumPy 배열로부터 생성:\n{tensor_from_numpy}")

# 2. 텐서 연산 (Tensor Operations)
# NumPy와 유사하게 연산이 가능합니다.
print("\n--- 텐서 연산 ---")
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# 텐서 덧셈 (Element-wise addition)
c = a + b
print(f"덧셈 결과:\n{c}")

# 행렬 곱셈 (Matrix multiplication)
d = tf.matmul(a, b)
print(f"행렬 곱셈 결과:\n{d}")

# 3. 자동 미분 (Autograd)
# tf.GradientTape를 사용하여 미분 연산을 추적할 수 있습니다.
print("\n--- 자동 미분 (GradientTape) ---")
# 'tf.Variable'을 사용해야 미분 대상이 됩니다.
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    # y = x^2 연산을 추적합니다.
    y = x * x

# y를 x에 대해 미분합니다. (dy/dx = 2x)
dy_dx = tape.gradient(y, x)
print(f"dy/dx (x=3) = {dy_dx.numpy()}")  # 2 * 3 = 6
