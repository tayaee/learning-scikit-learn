import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# 1. 데이터셋 로드 및 전처리 (CNN 예제와 동일)
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

# 2. 모델 구축 (CNN 예제와 동일)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10),
])

# 3. 콜백 설정
# 과적합을 방지하기 위한 EarlyStopping 콜백
# 'val_loss'가 5 에포크 동안 개선되지 않으면 학습을 조기 종료합니다.
early_stopping_callback = EarlyStopping(
    monitor="val_loss",  # 관찰할 지표
    patience=5,  # 지표가 개선되지 않아도 기다릴 에포크 수
    restore_best_weights=True,  # 학습 종료 시, 가장 성능이 좋았던 가중치로 복원
)

# 최적의 모델 가중치를 저장하는 ModelCheckpoint 콜백
# 'val_loss'가 가장 낮은 에포크의 모델만 저장합니다.
checkpoint_filepath = "best_model.h5"
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,  # 가중치만 저장
    monitor="val_loss",
    mode="min",  # val_loss가 최소가 될 때 저장
    save_best_only=True,  # 가장 좋은 모델만 저장
)

# 4. 모델 컴파일 및 학습
model.compile(
    optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
)

# callbacks 인수에 설정한 콜백들을 리스트 형태로 전달합니다.
history = model.fit(
    train_images,
    train_labels,
    epochs=20,  # 충분히 많은 에포크를 설정하여 EarlyStopping이 작동하도록 유도
    validation_data=(test_images, test_labels),
    callbacks=[early_stopping_callback, model_checkpoint_callback],
)

# 5. 최적의 가중치 불러와서 모델 평가
# EarlyStopping 덕분에 이미 최적의 가중치가 복원된 상태이지만,
# 명시적으로 저장된 파일을 불러와서 사용할 수도 있습니다.
model.load_weights(checkpoint_filepath)
loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f"\n최적의 가중치로 복원된 모델 정확도: {accuracy:.4f}")

# 저장된 파일 삭제
os.remove(checkpoint_filepath)
print(f"\n저장된 파일 '{checkpoint_filepath}'를 삭제했습니다.")
