import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers
import numpy as np

# 1. 가상 데이터셋 생성
# 실제 이미지 데이터셋이 없으므로, (224, 224, 3) 크기의 가상 데이터를 만듭니다.
# ResNet50 모델의 입력 크기에 맞춥니다.
(X_train, y_train), (X_test, y_test) = (
    (np.random.rand(100, 224, 224, 3), np.random.randint(0, 10, 100)),
    (np.random.rand(20, 224, 224, 3), np.random.randint(0, 10, 20)),
)

# 2. 사전 학습된 모델 불러오기
# ResNet50 모델을 불러옵니다. 'include_top=False'로 설정하여 마지막 분류 레이어는 제외합니다.
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# 3. 모델 동결 (Freeze)
# 사전 학습된 모델의 가중치를 업데이트하지 않도록 동결합니다.
base_model.trainable = False

# 4. 새로운 분류기 층 추가
# ResNet 모델 위에 새로운 층을 추가하여 커스텀 데이터셋에 맞춥니다.
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)  # 전역 평균 풀링으로 특성 벡터 생성
x = layers.Dense(128, activation="relu")(x)
predictions = layers.Dense(10, activation="softmax")(x)  # 10개 클래스 분류를 위한 출력 층

# 새로운 모델 정의
model = models.Model(inputs=base_model.input, outputs=predictions)

# 5. 모델 컴파일 및 학습
# 동결된 모델은 학습에서 제외됩니다.
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# 모델 학습 (가상 데이터 사용)
model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))

# 모델 구조 요약
model.summary()
