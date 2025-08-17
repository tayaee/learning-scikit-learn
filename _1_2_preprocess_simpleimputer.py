"""
전처리 (Preprocessing):
    데이터를 모델이 학습하기에 적합한 형태로 변환합니다. 결측값 처리, 데이터 스케일링, 특징 추출 등이 포함됩니다.
    예: sklearn.preprocessing.StandardScaler, sklearn.impute.SimpleImputer

SimpleImputer:
    SimpleImputer는 데이터 전처리 단계에서 결측값(missing values)을 채우는 데 사용되는 변환기(Transformer)입니다.
    결측값을 특정 전략(예: 평균, 중앙값, 최빈값 등)을 사용하여 채움으로써,
    결측값이 있는 데이터를 머신러닝 모델이 학습할 수 있는 형태로 만듭니다.
"""

# 1. 필요한 라이브러리 임포트
import numpy as np
import pandas as pd  # 데이터프레임으로 데이터를 다루기 위해 사용
from sklearn.impute import SimpleImputer  # SimpleImputer 변환기
from sklearn.model_selection import train_test_split  # 데이터 분할

# 2. 예제 데이터 생성
# 결측값(np.nan)이 포함된 데이터를 만듭니다.
# 'Age' 컬럼은 수치형, 'Embarked' 컬럼은 범주형으로 가정합니다.
data = {
    "Age": [20, 25, 30, np.nan, 40, 45, 50, np.nan],
    "Fare": [100, 150, np.nan, 200, 250, 300, 350, 400],
    "Embarked": ["S", "C", "S", "Q", np.nan, "S", "C", "S"],
}
df = pd.DataFrame(data)

print("원본 데이터프레임:")
print(df)

# 3. 수치형 데이터와 범주형 데이터 분리
# SimpleImputer는 기본적으로 수치형 데이터를 처리하는 데 사용됩니다.
# 범주형 데이터는 다른 전략이 필요할 수 있으므로 분리합니다.
numeric_data = df[["Age", "Fare"]]
categorical_data = df[["Embarked"]]

# 4. 수치형 데이터에 대한 SimpleImputer 적용
# SimpleImputer 추정기(변환기) 인스턴스 생성
# strategy='mean': 결측값을 해당 특징의 평균값으로 채웁니다.
imputer_numeric = SimpleImputer(strategy="mean")

# fit_transform: 학습 데이터에 대한 평균을 계산(fit)하고 결측값을 채웁니다(transform).
numeric_data_imputed = imputer_numeric.fit_transform(numeric_data)

# 5. 범주형 데이터에 대한 SimpleImputer 적용
# strategy='most_frequent': 결측값을 해당 특징의 최빈값(가장 자주 등장하는 값)으로 채웁니다.
imputer_categorical = SimpleImputer(strategy="most_frequent")
categorical_data_imputed = imputer_categorical.fit_transform(categorical_data)

# 6. 결과 확인
print("\n결측값이 채워진 수치형 데이터:")
print(pd.DataFrame(numeric_data_imputed, columns=["Age", "Fare"]))
print("\n결측값이 채워진 범주형 데이터:")
print(pd.DataFrame(categorical_data_imputed, columns=["Embarked"]))

# 7. (보너스) 학습/테스트 세트 분리 후 Imputer 적용 예시
# 데이터 분할 후 Imputer 적용
X_train, X_test = train_test_split(numeric_data, test_size=0.3, random_state=42)

# 학습 데이터로 fit (평균 계산)
imputer_split = SimpleImputer(strategy="mean")
imputer_split.fit(X_train)

# 학습 데이터는 fit_transform으로 변환
X_train_imputed = imputer_split.transform(X_train)

# 테스트 데이터는 transform으로만 변환 (학습 데이터의 평균 사용)
X_test_imputed = imputer_split.transform(X_test)

print("\n학습 데이터의 평균:", imputer_split.statistics_)
print("\n결측값이 채워진 학습 데이터:")
print(pd.DataFrame(X_train_imputed, columns=["Age", "Fare"]))
print("\n결측값이 채워진 테스트 데이터:")
print(pd.DataFrame(X_test_imputed, columns=["Age", "Fare"]))

"""
원본 데이터프레임:
    Age   Fare Embarked
0  20.0  100.0        S
1  25.0  150.0        C
2  30.0    NaN        S
3   NaN  200.0        Q
4  40.0  250.0      NaN
5  45.0  300.0        S
6  50.0  350.0        C
7   NaN  400.0        S

결측값이 채워진 수치형 데이터:
    Age   Fare
0  20.0  100.0
1  25.0  150.0
2  30.0  250.0
3  35.0  200.0
4  40.0  250.0
5  45.0  300.0
6  50.0  350.0
7  35.0  400.0

결측값이 채워진 범주형 데이터:
  Embarked
0        S
1        C
2        S
3        Q
4        S
5        S
6        C
7        S

학습 데이터의 평균: [ 40. 300.]

결측값이 채워진 학습 데이터:
    Age   Fare
0  40.0  400.0
1  30.0  300.0
2  40.0  250.0
3  40.0  200.0
4  50.0  350.0

결측값이 채워진 테스트 데이터:
    Age   Fare
0  25.0  150.0
1  45.0  300.0
2  20.0  100.0
"""

"""
Q1. SimpleImputer는 어떤 문제를 해결하기 위해 사용되나요?

    - SimpleImputer는 데이터셋에 존재하는 **결측값(missing values)**을 채우는 문제를 해결하기 위해 사용됩니다. 
      대부분의 머신러닝 알고리즘은 결측값이 포함된 데이터를 처리할 수 없으므로, 모델 학습 전에 결측값을 적절한 값으로 대체해야 합니다.

Q2. strategy 매개변수의 주요 옵션들은 무엇이며, 각각 어떤 상황에 적합한가요?

    - 'mean': 결측값을 해당 특징의 평균값으로 채웁니다. 데이터가 정규분포를 따르거나, 이상치(outlier)의 영향이 크지 않을 때 적합합니다.
    - 'median': 결측값을 중앙값으로 채웁니다. 이상치가 많아 평균값이 왜곡될 가능성이 있을 때 더 안정적인 선택입니다.
    - 'most_frequent': 결측값을 해당 특징의 **최빈값(가장 자주 등장하는 값)**으로 채웁니다. 수치형 데이터와 범주형 데이터 모두에 사용할 수 있습니다.
    - 'constant': 결측값을 사용자가 지정한 상수로 채웁니다.

Q3. 코드에서 수치형 데이터에는 strategy='mean'을, 범주형 데이터에는 strategy='most_frequent'를 사용했습니다. 이렇게 다른 전략을 적용하는 이유는 무엇인가요?

    - 수치형 데이터(Age, Fare)는 연속적인 값으로, 평균이나 중앙값 같은 통계적 지표를 계산하는 것이 의미가 있습니다.
    - 범주형 데이터(Embarked)는 이산적인 클래스(문자열)로 구성되어 있어 평균을 계산할 수 없습니다. 
      따라서 'S', 'C', 'Q'와 같은 범주 중에서 가장 많이 등장하는 값인 최빈값으로 결측값을 채우는 것이 자연스러운 접근 방식입니다.

Q4. SimpleImputer를 train_test_split으로 나뉜 학습 데이터와 테스트 데이터에 적용할 때, 
    fit()과 transform() 메서드를 어떻게 사용해야 하나요? 이 과정이 중요한 이유는 무엇인가요?

    - 학습 데이터: fit_transform() 또는 fit() 후 transform()을 사용하여 학습 데이터의 통계량을 계산하고 결측값을 채워야 합니다.
    - 테스트 데이터: transform()만 사용해야 합니다.
    - 이 과정이 중요한 이유는 **데이터 누수(Data Leakage)**를 방지하기 위함입니다. 
      학습 데이터의 통계량(예: 평균)을 사용해 결측값을 채우고, 
      테스트 데이터의 결측값은 학습 데이터의 통계량으로 채워야 합니다. 
      만약 테스트 데이터의 통계량을 별도로 계산(fit)하여 사용하면, 
      학습 모델이 실제로는 알 수 없는 정보(테스트 데이터의 평균)를 미리 알게 되어 모델 성능이 과대평가되는 문제가 발생합니다.
      pandas의 fillna로 같은 결과를 구현할 때 주의해야 할 사항은 아래와 같습니다.
      예를 들어 mean 값으로 결측치를 채우기로 했다면
      mean 값은 training data로부터 계산된 후, training data와 test data에 적용되어야 합니다.
      입력값 전체 (train + test)로부터 mean을 계산한 후, fillna를 입력값 전체에 대해 적용하면
      test data에 대한 정보가 누수되어 모델이 과대평가될 수 있습니다. 

"""
