"""
ColumnTransformer는 데이터셋의 여러 컬럼에 각각 다른 전처리 변환기를 적용할 수 있게 해주는 도구입니다.
이를 Pipeline과 함께 사용하면, 복합적인 전처리 과정을 하나의 워크플로우로 통합하여
훨씬 더 강력하고 실용적인 머신러닝 파이프라인을 구축할 수 있습니다.
"""

# 1. 필요한 라이브러리 임포트
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # 여러 컬럼에 다른 전처리를 적용
from sklearn.ensemble import RandomForestClassifier  # 최종 모델
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline  # 파이프라인 API
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 2. 예제 데이터 생성 (다양한 데이터 타입 포함)
# 수치형 특징과 범주형 특징이 섞인 실제와 유사한 데이터셋을 만듭니다.
data = {
    "age": np.random.randint(18, 65, 100),
    "fare": np.random.uniform(50, 500, 100),
    "embarked": np.random.choice(["S", "C", "Q"], 100, p=[0.7, 0.2, 0.1]),
    "sex": np.random.choice(["male", "female"], 100, p=[0.6, 0.4]),
    "target": np.random.randint(0, 2, 100),  # 이진 분류를 위한 타겟 변수
}
df = pd.DataFrame(data)
print(df)
X = df.drop("target", axis=1)
print(X)
y = df["target"]
print(y)

# 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("원본 데이터셋의 일부:")
print(X_train.head())
print("-" * 50)


# ==============================================================================
# 섹션 1: ColumnTransformer를 이용한 전처리 파이프라인 구축
# 각 데이터 타입에 맞는 전처리기를 정의하고, ColumnTransformer로 통합합니다.
# ==============================================================================

# 수치형 특징과 범주형 특징 구분
numerical_features = ["age", "fare"]
categorical_features = ["embarked", "sex"]

# (1) 수치형 데이터 전처리를 위한 파이프라인 (스케일링)
# StandardScaler를 사용합니다.
numerical_pipeline = Pipeline([("scaler", StandardScaler())])

# (2) 범주형 데이터 전처리를 위한 파이프라인 (원-핫 인코딩)
# drop='first'를 사용하여 다중공선성을 방지합니다.
categorical_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

# (3) ColumnTransformer를 사용하여 위 파이프라인들을 통합
# 'age'와 'fare'에는 스케일러 적용
# 'embarked'와 'sex'에는 원-핫 인코더 적용
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ],
    remainder="passthrough",  # 위에서 지정하지 않은 컬럼은 그대로 둡니다. (이 예제에서는 해당 없음)
)

print("ColumnTransformer를 포함한 전처리 파이프라인 구성 완료.")
print("-" * 50)

# ==============================================================================
# 섹션 2: 전체 워크플로우를 Pipeline으로 통합
# 전처리 단계와 최종 모델을 하나의 객체로 묶습니다.
# 첫 번째 단계: ColumnTransformer를 사용한 전처리
# 두 번째 단계: 최종 분류 모델
# ==============================================================================
full_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# Pipeline.fit() 호출 한 번으로 모든 전처리와 모델 학습이 자동화됩니다.
print("전체 파이프라인 학습 시작...")
full_pipeline.fit(X_train, y_train)
print("전체 파이프라인 학습 완료!")
print("-" * 50)


# ==============================================================================
# 섹션 3: 파이프라인을 사용한 예측 및 평가
# ==============================================================================
# Pipeline.predict() 호출 한 번으로 전처리와 예측이 자동화됩니다.
y_pred = full_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"테스트 데이터셋 정확도: {accuracy:.4f}")
print("파이프라인을 사용하면 전처리 단계와 모델 학습 단계가 자동으로 순차적으로 실행됩니다.")

# 파이프라인을 이용한 교차 검증
scores = cross_val_score(full_pipeline, X, y, cv=5, scoring="accuracy")
print(f"\n파이프라인을 이용한 5-폴드 교차 검증 점수: {scores}")
print(f"교차 검증 평균 정확도: {np.mean(scores):.4f}")
print("-" * 50)

"""
원본 데이터셋의 일부:
    age        fare embarked     sex
11   22  101.817375        C  female
47   54  318.720087        C  female
85   34   85.674513        S  female
28   53   87.227896        S  female
93   58  368.277939        S  female
--------------------------------------------------
ColumnTransformer를 포함한 전처리 파이프라인 구성 완료.
--------------------------------------------------
전체 파이프라인 학습 시작...
전체 파이프라인 학습 완료!
--------------------------------------------------
테스트 데이터셋 정확도: 0.5000
파이프라인을 사용하면 전처리 단계와 모델 학습 단계가 자동으로 순차적으로 실행됩니다.

파이프라인을 이용한 5-폴드 교차 검증 점수: [0.55 0.3  0.45 0.55 0.45]
교차 검증 평균 정확도: 0.4600
--------------------------------------------------
"""

"""
복습

Q1. ColumnTransformer를 사용하지 않고, 여러 전처리기(예: StandardScaler, OneHotEncoder)를 
    개별적으로 적용할 때 어떤 문제점이 발생할 수 있나요?

    코드 복잡성 증가: 
        각 컬럼 그룹에 대해 fit_transform과 transform을 수동으로 호출해야 합니다. 
        이로 인해 코드가 길어지고 가독성이 떨어집니다.
    오류 발생 가능성: 
        학습 데이터의 통계량(평균, 최댓값 등)을 기억하고 테스트 데이터에 적용해야 하는데, 
        이 과정에서 실수가 발생하기 쉽습니다.
    데이터 누수 위험: 
        특히 교차 검증 시 각 폴드마다 학습 데이터에만 fit을 수행해야 하는데, 
        수동으로 처리하면 테스트 데이터의 정보가 학습 과정에 유입되는 
        데이터 누수(Data Leakage)가 발생할 수 있습니다.

Q2. 데모 코드에서 preprocessor = ColumnTransformer(...) 코드의 
    transformers 매개변수에 전달된 튜플 ('num', numerical_pipeline, numerical_features)의 
    각 요소는 어떤 역할을 하나요?

    'num': 
        이 변환기의 이름입니다. 
        고유한 이름을 지정하여 나중에 파이프라인에서 이 단계에 접근하거나 하이퍼파라미터 튜닝 시 사용합니다.

    numerical_pipeline: 
        이 컬럼들에 적용할 변환기(Transformer) 인스턴스입니다. 
        데모에서는 StandardScaler를 포함한 Pipeline 객체입니다.
    
    numerical_features: 
        이 변환기를 적용할 컬럼들의 리스트입니다. 
        데모에서는 ['age', 'fare']가 해당됩니다.

Q3. ColumnTransformer와 Pipeline을 결합했을 때 얻을 수 있는 가장 큰 이점은 무엇인가요?
    
    단일 워크플로우: 
        복잡한 전처리 과정과 모델 학습을 full_pipeline.fit() 
        한 번으로 모두 실행할 수 있어 코드가 매우 간결해집니다.
    
    자동화 및 안정성: 
        전처리 단계가 자동으로 순차적으로 실행되므로, 
        수동으로 데이터를 변환할 때 발생할 수 있는 휴먼 에러를 방지할 수 있습니다.
    
    데이터 누수 방지: 
        교차 검증 시 Pipeline이 자동으로 각 폴드의 학습 데이터에만 fit을 수행하므로, 
        테스트 데이터의 정보가 학습 과정에 유입되는 것을 원천적으로 차단합니다.

Q4. Pipeline을 교차 검증(cross_val_score)에 사용했을 때, ColumnTransformer는 각 폴드마다 어떻게 동작하나요?
    
    Pipeline은 교차 검증의 각 폴드(fold)가 생성될 때마다 
    내부적으로 ColumnTransformer를 포함한 전체 파이프라인 객체를 새롭게 초기화하고 학습합니다.
    
    즉, 매 폴드마다 ColumnTransformer는 해당 폴드의 학습 데이터(X_train에 해당)에만
    fit 메서드를 호출하여 통계량(평균, 표준편차 등)을 계산합니다.
    
    이후 테스트 데이터에는 이 학습 데이터로부터 얻은 통계량으로 transform만 적용됩니다.
    이 과정을 통해 교차 검증이 엄격하게 수행되며, 테스트 데이터의 정보가 학습에 영향을 주지 않도록 보장합니다.
"""
