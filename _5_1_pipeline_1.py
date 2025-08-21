"""
파이프라인 (Pipeline) API
    데이터 전처리부터 모델 학습까지 여러 단계를 하나의 객체로 묶어 관리할 수 있도록 해줍니다.
    Pipeline은 코드의 가독성을 높이고, 전처리 단계에서 발생할 수 있는 데이터 누수(data leakage)를 방지하는 데 매우 유용합니다.
    scikit-learn의 이러한 일관성 있고 풍부한 API는 머신러닝 모델을 쉽게 개발하고, 유지보수하며, 확장할 수 있도록 도와줍니다.

Pipeline:
    Pipeline은 여러 전처리 단계와 최종 모델을 하나의 객체로 묶어주는 강력한 도구입니다.
    이 데모 코드는 데이터 스케일링(전처리)과 모델 학습을 Pipeline으로 통합하여,
    Pipeline이 어떻게 코드의 간결성을 높이고, 특히 교차 검증에서 데이터 누수(Data Leakage)를 효과적으로 방지하는지 보여줍니다.
"""

# 1. 필요한 라이브러리 임포트
import numpy as np
from sklearn.datasets import load_iris  # 예제 데이터셋
from sklearn.linear_model import LogisticRegression  # 최종 모델
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline  # Pipeline API
from sklearn.preprocessing import StandardScaler  # 데이터 전처리 (스케일러)

# 2. 예제 데이터 로드 및 분할
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==============================================================================
# 섹션 1: Pipeline을 사용하지 않는 경우 (수동 구현)
# 전처리와 모델 학습을 각각 분리하여 실행합니다.
# ==============================================================================
print("=== [Section 1] Pipeline을 사용하지 않는 경우 ===")

# 전처리 (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 테스트 데이터는 학습 데이터의 통계량으로 변환

# 모델 학습 (LogisticRegression)
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 예측 및 평가
y_pred = model.predict(X_test_scaled)
accuracy_manual = accuracy_score(y_test, y_pred)
print(f"수동 구현 방식의 정확도: {accuracy_manual:.4f}")
print("-" * 50)


# ==============================================================================
# 섹션 2: Pipeline을 사용하는 경우
# 전처리와 모델을 하나의 객체로 묶어 관리합니다.
# ==============================================================================
print("=== [Section 2] Pipeline을 사용하는 경우 ===")

# Pipeline 객체 생성
# steps: 튜플 리스트로 구성됩니다. (단계 이름, 추정기 인스턴스)
# 각 단계는 '변환기(Transformer)' 또는 '최종 추정기(Estimator)'여야 합니다.
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # 첫 번째 단계: 데이터 표준화
    (
        "classifier",
        LogisticRegression(random_state=42),
    ),  # 두 번째 단계: 로지스틱 회귀 모델
])

# pipeline.fit() 호출 한 번으로 전처리, 학습이 모두 자동화됩니다.
# 내부적으로 scaler.fit_transform() 후 classifier.fit()이 순서대로 실행됩니다.
pipeline.fit(X_train, y_train)

# pipeline.predict() 호출 한 번으로 전처리, 예측이 모두 자동화됩니다.
# 내부적으로 scaler.transform() 후 classifier.predict()이 순서대로 실행됩니다.
y_pred_pipeline = pipeline.predict(X_test)
accuracy_pipeline = accuracy_score(y_test, y_pred_pipeline)
print(f"Pipeline을 사용한 정확도: {accuracy_pipeline:.4f}")
print("-" * 50)


# ==============================================================================
# 섹션 3: Pipeline과 교차 검증 (Cross-Validation)
# Pipeline이 데이터 누수를 어떻게 방지하는지 보여줍니다.
# ==============================================================================
print("=== [Section 3] Pipeline과 교차 검증 ===")

# Pipeline을 cross_val_score 함수에 전달합니다.
# cross_val_score는 각 폴드마다 Pipeline을 새로 생성하고 학습합니다.
# 따라서 매번 학습 데이터로만 스케일링 통계량을 계산합니다.
scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

print(f"Pipeline을 사용한 5-폴드 교차 검증 점수: {scores}")
print(f"Pipeline 교차 검증 평균 정확도: {np.mean(scores):.4f}")
print("교차 검증 시, Pipeline은 각 폴드의 학습 데이터로만 전처리합니다. (데이터 누수 방지)")
print("-" * 50)

"""
=== [Section 1] Pipeline을 사용하지 않는 경우 ===
수동 구현 방식의 정확도: 1.0000
--------------------------------------------------
=== [Section 2] Pipeline을 사용하는 경우 ===
Pipeline을 사용한 정확도: 1.0000
--------------------------------------------------
=== [Section 3] Pipeline과 교차 검증 ===
Pipeline을 사용한 5-폴드 교차 검증 점수: [0.96666667 1.         0.93333333 0.9        1.        ]
Pipeline 교차 검증 평균 정확도: 0.9600
교차 검증 시, Pipeline은 각 폴드의 학습 데이터로만 전처리합니다. (데이터 누수 방지)
"""

"""
복습

Q1. Pipeline을 사용하는 가장 큰 장점은 무엇인가요? 섹션 1과 2의 코드를 비교하여 설명해 주세요.

    Pipeline의 가장 큰 장점은 전처리 단계와 모델 학습 단계를 
    하나의 객체로 통합하여 코드를 간결하게 만들고 워크플로우를 자동화하는 것입니다.

    섹션 1(수동 구현): 
        StandardScaler와 LogisticRegression을 별도로 선언하고, 
        fit_transform(), transform(), fit(), predict()를 각각 순서대로 호출해야 합니다. 
        이는 코드를 복잡하게 만듭니다.
    
    섹션 2(Pipeline): 
        Pipeline 객체 하나만 생성한 후 fit(), 
        predict() 메서드를 호출하면 모든 전처리 및 학습 과정이 내부적으로 자동으로 처리됩니다. 
        이는 코드의 가독성과 유지보수성을 크게 향상시킵니다.

Q2. pipeline.fit(X_train, y_train)이 내부적으로 어떤 순서로 동작하는지 설명해 주세요.
    
    pipeline.fit()이 호출되면, Pipeline은 steps에 정의된 순서대로 각 단계를 실행합니다.
    
        첫 번째 단계(scaler): 
            StandardScaler 객체의 fit_transform(X_train) 메서드가 호출되어 학습 데이터의 평균과 표준편차를 계산하고(fit), 
            데이터를 표준화(transform)합니다. 
            그 결과는 다음 단계의 입력으로 전달됩니다.
        
        두 번째 단계(classifier): 
            표준화된 데이터와 y_train을 인수로 받아 
            LogisticRegression 객체의 fit() 메서드가 호출되어 모델을 학습합니다.

Q3. **Pipeline이 교차 검증(Cross-Validation) 과정에서 **데이터 누수(Data Leakage)를 방지하는 원리를 설명해 주세요.
    
    cross_val_score(pipeline, ...)와 같이 Pipeline을 교차 검증에 사용하면, 
    Pipeline 객체는 각 폴드(fold)마다 새로 생성되고 학습됩니다.
    
    즉, 매 폴드마다 학습 데이터로만 scaler.fit_transform()이 실행되고, 
    테스트 데이터에는 scaler.transform()만 적용됩니다.
    
    이렇게 되면 테스트 데이터의 정보(평균, 표준편차)가 학습 과정에 절대로 유입되지 않으므로, 
    데이터 누수를 효과적으로 방지하고 모델의 일반화 성능을 객관적으로 평가할 수 있습니다.

Q4. Pipeline의 steps 매개변수에 전달되는 튜플 리스트의 각 요소 ('scaler', StandardScaler())에서 'scaler'와 StandardScaler()는 각각 어떤 역할을 하나요?
    
    StandardScaler()는 Pipeline의 한 단계로 사용될 추정기(Estimator) 인스턴스입니다. 
    이 인스턴스가 실제 전처리 또는 학습 작업을 수행합니다.
    
    'scaler'는 해당 단계의 **이름(name)**입니다. 
    이 이름은 Pipeline 내부에서 해당 단계의 추정기에 접근하거나, 
    하이퍼파라미터 튜닝(예: GridSearchCV) 시 매개변수를 지정하는 데 사용됩니다. 
    예를 들어 GridSearchCV에서 scaler__with_mean과 같은 형태로 매개변수를 지정할 수 있습니다.
"""
