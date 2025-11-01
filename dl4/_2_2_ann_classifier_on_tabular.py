import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import pandas as pd

# 1. 가상 데이터셋 생성
# 예제를 위해 10개의 특성(feature)을 가진 이진 분류 데이터셋을 만듭니다.
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
df["target"] = y

# 2. 데이터 전처리
# 신경망 학습을 위해 데이터를 표준화(Standardization)합니다.
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 훈련, 검증, 테스트 데이터셋 분리
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 3. ANN 모델 구축
# tf.keras.Sequential API를 사용해 모델을 순차적으로 쌓아 올립니다.
model = tf.keras.Sequential([
    # 입력 레이어: 입력 데이터의 특성 수(10개)와 동일하게 설정합니다.
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    # 히든 레이어: 중간에 데이터를 처리하는 레이어
    tf.keras.layers.Dense(32, activation="relu"),
    # 출력 레이어: 이진 분류이므로 sigmoid 활성화 함수를 사용합니다.
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

# 4. 모델 컴파일
# 옵티마이저, 손실 함수, 평가 지표를 설정합니다.
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",  # 이진 분류에 적합한 손실 함수
    metrics=["accuracy"],
)

# 5. 모델 학습
# 훈련 데이터로 모델을 학습시킵니다.
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1,
)

# 6. 모델 평가
# 테스트 데이터셋으로 모델의 성능을 최종 평가합니다.
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n테스트 데이터셋 정확도: {accuracy:.4f}")
