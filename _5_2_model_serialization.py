"""
학습된 모델을 파일로 저장하고, 필요할 때 다시 불러와서 사용하거나 배포하는 기능입니다. 모델을 매번 다시 학습시킬 필요 없이 효율적으로 사용할 수 있습니다.
    joblib 또는 pickle: joblib.dump() 함수로 모델을 파일로 저장하고, joblib.load()로 다시 불러옵니다.

joblib은 scikit-learn 모델과 같이 NumPy 배열을 포함하는 대용량 객체를 효율적으로 저장하고 로드하는 데 최적화된 라이브러리입니다.
모델을 학습한 후 다시 학습할 필요 없이 저장된 모델 파일을 재사용할 수 있어, 시간과 자원을 크게 절약하고 모델을 배포하는 데 필수적입니다.
"""

# 1. 필요한 라이브러리 임포트
import joblib  # 모델 저장/로드에 사용
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 2. 예제 데이터 생성 및 모델 학습
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 인스턴스 생성 및 학습
model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
print("모델 학습을 시작합니다...")
model.fit(X_train, y_train)
print("모델 학습 완료!")

# 학습된 모델의 성능을 테스트 데이터로 확인
initial_y_pred = model.predict(X_test)
initial_accuracy = accuracy_score(y_test, initial_y_pred)
print(f"학습 직후 테스트 데이터 정확도: {initial_accuracy:.4f}")
print("-" * 50)


# ==============================================================================
# 섹션 1: 학습된 모델을 파일로 저장하기
# joblib.dump() 함수를 사용하여 모델 객체를 파일로 직렬화(serialization)합니다.
# ==============================================================================
print("=== [Section 1] 모델 저장하기 ===")

# 저장할 파일명 정의 (일반적으로 확장자는 .joblib 또는 .pkl 사용)
model_filename = "trained_svc_model.joblib"

# joblib.dump(저장할 객체, 파일명)
joblib.dump(model, model_filename)
print(f"모델이 '{model_filename}' 파일로 성공적으로 저장되었습니다.")
print("-" * 50)


# ==============================================================================
# 섹션 2: 저장된 모델 파일을 불러와서 사용하기
# joblib.load() 함수를 사용하여 저장된 모델 객체를 메모리로 역직렬화(deserialization)합니다.
# ==============================================================================
print("=== [Section 2] 모델 불러와서 사용하기 ===")

# joblib.load(파일명)
# 새로운 변수에 저장된 모델을 불러옵니다.
loaded_model = joblib.load(model_filename)

# 불러온 모델의 예측 성능을 다시 확인
# 모델을 다시 학습할 필요 없이 바로 예측에 사용할 수 있습니다.
loaded_y_pred = loaded_model.predict(X_test)
loaded_accuracy = accuracy_score(y_test, loaded_y_pred)

print(f"불러온 모델의 테스트 데이터 정확도: {loaded_accuracy:.4f}")

# 저장 전후 모델의 정확도가 동일함을 확인
print(f"정확도 일치 여부: {initial_accuracy == loaded_accuracy}")
print("-" * 50)

# 팁:
# joblib.dump()의 compress 옵션을 사용하면 파일 크기를 줄일 수 있습니다.
joblib.dump(model, "compressed_model.joblib", compress=3)

"""
모델 학습을 시작합니다...
모델 학습 완료!
학습 직후 테스트 데이터 정확도: 0.8333
--------------------------------------------------
=== [Section 1] 모델 저장하기 ===
모델이 'trained_svc_model.joblib' 파일로 성공적으로 저장되었습니다.
--------------------------------------------------
=== [Section 2] 모델 불러와서 사용하기 ===
불러온 모델의 테스트 데이터 정확도: 0.8333
정확도 일치 여부: True
--------------------------------------------------
"""

"""
Q1. 모델 직렬화(Serialization)의 주요 목적은 무엇이며, 이것이 머신러닝 워크플로우에서 왜 중요한가요?
    
    모델 직렬화의 주요 목적은 학습된 머신러닝 모델의 객체 상태를 파일로 변환하여 영구적으로 저장하는 것입니다.
    이는 다음과 같은 이유로 중요합니다:
        재사용성: 한 번 학습된 모델을 필요할 때마다 다시 학습할 필요 없이 불러와서 바로 예측에 사용할 수 있습니다.
        배포: 학습된 모델을 웹 서비스나 애플리케이션에 쉽게 통합하여 배포할 수 있습니다.
        시간 및 자원 절약: 대규모 데이터셋으로 모델을 학습하는 데 드는 막대한 시간과 컴퓨팅 자원을 절약할 수 있습니다.

Q2. joblib.dump() 함수와 joblib.load() 함수는 각각 어떤 역할을 수행하며, 두 함수의 인자(parameter)는 무엇인가요?
    
    joblib.dump(value, filename, compress=0, ...):
        역할: 파이썬 객체(value)를 파일(filename)로 직렬화하여 저장합니다.
        주요 인자:
            value: 저장하고자 하는 파이썬 객체(예: 학습된 scikit-learn 모델).
            filename: 객체를 저장할 파일의 경로와 이름.
    
    joblib.load(filename, mmap_mode=None):
        역할: 지정된 파일(filename)에서 직렬화된 객체를 읽어와 메모리로 복원(역직렬화)합니다.
        주요 인자:
            filename: 불러올 파일의 경로와 이름.

Q3. 데모 코드에서 joblib을 사용했을 때, 모델을 다시 학습할 필요 없이 정확도가 동일하게 나온 이유는 무엇인가요?
    
    joblib.dump()는 학습된 모델 객체의 **모든 상태(State)**를 그대로 파일에 저장합니다. 
    여기에는 모델의 하이퍼파라미터(kernel, C, gamma)뿐만 아니라, 
    학습 과정에서 결정된 가중치(weights)나 서포트 벡터(SVC의 경우) 등 모델의 예측에 필요한 모든 정보가 포함됩니다.
    
    joblib.load()로 이 파일을 다시 불러오면, 저장되었던 모델의 상태가 완벽하게 복원됩니다. 
    따라서 모델을 다시 학습하지 않아도 학습 직후와 동일한 예측 성능을 보입니다.

Q4. joblib이 pickle보다 scikit-learn 모델을 저장하는 데 더 선호되는 이유는 무엇인가요?

    효율성: 
        joblib은 pickle보다 특히 NumPy 배열을 포함한 대용량 객체를 다룰 때 훨씬 빠르고 효율적입니다. 
        scikit-learn의 모델들은 내부적으로 NumPy 배열을 사용하기 때문에 joblib에 최적화되어 있습니다.

    메모리 매핑: 
        joblib은 메모리 매핑 기술을 지원하여, 
        큰 모델 파일을 메모리에 한 번에 불러오지 않고 필요할 때마다 디스크에서 읽어올 수 있습니다. 
        이는 메모리 사용량을 최소화하는 데 큰 장점이 됩니다.

    안전성: 
        pickle은 신뢰할 수 없는 소스로부터 받은 파일에 대해 보안 위험이 있을 수 있습니다. 
        반면, joblib은 이와 관련하여 상대적으로 더 안전하다고 알려져 있습니다.
"""
