"""
모델 관리 (Model Management)
    학습된 모델을 파일로 저장하는 것을 넘어, 모델의 정보(메타데이터)를 함께 저장하여
    체계적으로 관리하고, 필요할 때 쉽게 검색하고 재사용할 수 있도록 하는 과정입니다.
    이는 MLOps의 중요한 첫걸음입니다.
"""

# 1. 필요한 라이브러리 임포트
import datetime
import json
import os
import uuid

import joblib
import sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def save_model_with_metadata(model, metrics, model_dir="models"):
    """
    학습된 모델과 관련 메타데이터를 함께 저장합니다.

    Args:
        model: 학습된 scikit-learn 모델 객체.
        metrics (dict): 모델의 성능 지표 (예: {'accuracy': 0.95}).
        model_dir (str): 모델과 메타데이터를 저장할 디렉토리.

    Returns:
        str: 저장된 모델의 고유 ID.
    """
    # 저장할 디렉토리가 없으면 생성
    os.makedirs(model_dir, exist_ok=True)

    # 1. 모델 식별 정보 생성
    model_id = str(uuid.uuid4())
    model_filename = f"{model_id}.joblib"
    metadata_filename = f"{model_id}.json"
    model_filepath = os.path.join(model_dir, model_filename)
    metadata_filepath = os.path.join(model_dir, metadata_filename)

    # 2. 모델 파일 저장 (직렬화)
    joblib.dump(model, model_filepath)
    print(f"모델이 '{model_filepath}' 경로에 저장되었습니다.")

    # 3. 메타데이터 딕셔너리 생성
    metadata = {
        "model_uuid": model_id,
        "model_filepath": model_filepath,
        "model_class": model.__class__.__name__,
        "hyperparameters": model.get_params(),
        "metrics": metrics,
        "timestamp": datetime.datetime.now().isoformat(),
        "sklearn_version": sklearn.__version__,
        "description": "SVC model for binary classification demo.",
    }

    # 4. 메타데이터 JSON 파일 저장
    with open(metadata_filepath, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"메타데이터가 '{metadata_filepath}' 경로에 저장되었습니다.")

    return model_id


# ==============================================================================
# 섹션 1: 모델 학습 및 메타데이터와 함께 저장하기
# ==============================================================================
print("=== [Section 1] 모델 학습 및 저장 ===")

# 예제 데이터 생성 및 모델 학습
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
model_metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
}

# 함수를 사용하여 모델과 메타데이터 저장
saved_model_id = save_model_with_metadata(model, model_metrics)
print(f"저장된 모델 ID: {saved_model_id}")
print("-" * 50)


# ==============================================================================
# 섹션 2: 메타데이터를 이용해 모델 불러오기
# ==============================================================================
print("\n=== [Section 2] 메타데이터를 이용한 모델 로드 및 사용 ===")

# 저장된 메타데이터 파일 경로
metadata_path_to_load = os.path.join("models", f"{saved_model_id}.json")

# 1. 메타데이터 로드
with open(metadata_path_to_load, "r") as f:
    loaded_metadata = json.load(f)

print("불러온 메타데이터:")
print(json.dumps(loaded_metadata, indent=4))

# 2. 메타데이터에서 모델 파일 경로를 찾아 모델 로드
model_path_to_load = loaded_metadata["model_filepath"]
loaded_model = joblib.load(model_path_to_load)

# 3. 불러온 모델로 예측 수행
new_prediction = loaded_model.predict(X_test[:1])
print(f"\n불러온 모델로 새로운 데이터 예측 결과: {new_prediction}")
print("-" * 50)
