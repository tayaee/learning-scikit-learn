import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 1. 데이터셋 로드 및 전처리
# MNIST 손글씨 이미지 데이터셋을 사용합니다.
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# 이미지를 픽셀값 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0

# Keras CNN 레이어에 맞게 이미지 차원을 (높이, 너비, 채널)로 변경합니다.
# 흑백 이미지이므로 채널은 1입니다.
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

# 2. CNN 모델 구축
model = models.Sequential()
# 첫 번째 합성곱 층: 32개의 3x3 필터를 사용합니다.
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
# 풀링 층: 2x2 크기로 이미지 크기를 줄입니다.
model.add(layers.MaxPooling2D((2, 2)))
# 두 번째 합성곱 층: 64개의 필터를 사용합니다.
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
# 세 번째 합성곱 층
model.add(layers.Conv2D(64, (3, 3), activation="relu"))

# 3. 모델 컴파일 및 학습
# 모델의 2차원 출력을 1차원으로 펼칩니다.
model.add(layers.Flatten())
# 밀집 연결(Dense) 층을 추가합니다.
model.add(layers.Dense(64, activation="relu"))
# 출력 층: 10개의 클래스(0-9)를 분류하기 위해 10개의 유닛을 사용합니다.
model.add(layers.Dense(10))

# 옵티마이저, 손실 함수, 평가 지표를 설정합니다.
model.compile(
    optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
)

# 모델 학습
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# 모델 구조 요약
model.summary()
