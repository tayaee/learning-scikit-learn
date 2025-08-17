"""
특징(Feature) 전처리 및 엔지니어링

모델 학습에 적합한 형태로 데이터를 가공하는 다양한 기술들입니다.
    sklearn.preprocessing.OneHotEncoder: 범주형 데이터를 원-핫 인코딩하여 머신러닝 모델이 이해할 수 있는 형태로 변환합니다. (예: '서울', '부산' -> [1, 0], [0, 1])
    sklearn.preprocessing.MinMaxScaler: 데이터를 0과 1 사이의 값으로 스케일링합니다. StandardScaler와는 또 다른 중요한 스케일링 방법입니다.
    sklearn.feature_extraction.text: 텍스트 데이터에서 특징을 추출하는 기능입니다. CountVectorizer나 TfidfVectorizer가 대표적입니다.

OneHotEncoder:
    OneHotEncoder는 머신러닝 모델이 학습할 수 있도록 범주형 데이터를 수치형 데이터로 변환하는 전처리 도구입니다.
    이 인코딩 방식은 각 범주(카테고리)를 독립적인 이진(0 또는 1) 특징으로 변환하여,
    범주 간의 순서나 관계를 모델이 잘못 해석하는 것을 방지하는 데 유용합니다.
"""

# 1. 필요한 라이브러리 임포트
import pandas as pd
from sklearn.compose import ColumnTransformer  # 여러 특징에 다른 변환기를 적용하기 위해 사용
from sklearn.preprocessing import OneHotEncoder

# 2. 예제 데이터 생성
# 'City'와 'Weather'라는 범주형 특징(Feature)을 가진 데이터프레임을 만듭니다.
# 'City'에는 3개의 고유한 범주, 'Weather'에는 2개의 고유한 범주가 있습니다.
data = {
    "City": ["Seoul", "Busan", "Seoul", "Jeju", "Busan"],
    "Weather": ["Sunny", "Rainy", "Sunny", "Sunny", "Rainy"],
    "Temperature": [25, 20, 26, 22, 21],
}
df = pd.DataFrame(data)

print("원본 데이터프레임:")
print(df)
print("-" * 50)

# 3. OneHotEncoder 인스턴스 생성 및 적용
# drop='first': 첫 번째 범주를 제외하여 다중공선성(multicollinearity) 문제를 피합니다.
#               (예: 'City_Busan', 'City_Seoul'이 모두 0이면 'City_Jeju'를 의미)
# handle_unknown='ignore': 학습 데이터에 없던 새로운 범주가 테스트 데이터에 나타나도 무시합니다.
# sparse_output=False: 희소 행렬이 아닌 일반 배열(ndarray)로 출력합니다.
encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)

# fit_transform()을 사용하여 'City' 컬럼을 인코딩합니다.
# encoder는 'Seoul', 'Busan', 'Jeju'의 순서를 학습합니다.
city_encoded = encoder.fit_transform(df[["City"]])

# 인코딩된 결과를 데이터프레임으로 변환하여 확인
city_encoded_df = pd.DataFrame(city_encoded, columns=encoder.get_feature_names_out(["City"]))
print("OneHotEncoder 적용 후 'City' 컬럼:")
print(city_encoded_df)
print("-" * 50)

# 4. ColumnTransformer를 이용한 여러 컬럼 동시 처리
# 실제 데이터에서는 여러 컬럼에 각각 다른 전처리를 적용해야 합니다.
# ColumnTransformer는 이 과정을 효율적으로 수행합니다.
# (1) OneHotEncoder를 적용할 컬럼 리스트 ('City', 'Weather')
categorical_features = ["City", "Weather"]

# (2) ColumnTransformer 인스턴스 생성
# transformers: (변환기 이름, 변환기 인스턴스, 적용할 컬럼 리스트)의 튜플 리스트
preprocessor = ColumnTransformer(
    transformers=[
        (
            "onehot",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            categorical_features,
        )
    ],
    remainder="passthrough",  # 'Temperature'와 같이 변환기가 필요 없는 컬럼은 그대로 둡니다.
)

# fit_transform()을 사용하여 여러 컬럼을 한 번에 전처리합니다.
X_processed = preprocessor.fit_transform(df)

# 최종 결과 확인
processed_df = pd.DataFrame(
    X_processed,
    # 인코딩된 컬럼 이름과 원본 컬럼 이름을 결합하여 새로운 컬럼 이름 리스트 생성
    columns=preprocessor.get_feature_names_out(),
)
print("ColumnTransformer를 이용한 전체 데이터프레임 전처리 결과:")
print(processed_df)

"""
실행 예:

원본 데이터프레임:
    City Weather  Temperature
0  Seoul   Sunny           25
1  Busan   Rainy           20
2  Seoul   Sunny           26
3   Jeju   Sunny           22
4  Busan   Rainy           21
--------------------------------------------------
OneHotEncoder 적용 후 'City' 컬럼:
   City_Jeju  City_Seoul
0        0.0         1.0
1        0.0         0.0        // City_Busan은 drop='first'로 제거됨
2        0.0         1.0
3        1.0         0.0
4        0.0         0.0        // City_Busan은 두 컬럼이 모두 0일 때를 의미
--------------------------------------------------
ColumnTransformer를 이용한 전체 데이터프레임 전처리 결과:
   onehot__City_Jeju  onehot__City_Seoul  onehot__Weather_Sunny  remainder__Temperature
0                0.0                 1.0                    1.0                    25.0
1                0.0                 0.0                    0.0                    20.0
2                0.0                 1.0                    1.0                    26.0
3                1.0                 0.0                    1.0                    22.0
4                0.0                 0.0                    0.0                    21.0
"""

"""
코드 복기

Q1. OneHotEncoder를 사용하는 주요 목적은 무엇이며, 범주형 데이터를 그대로 사용했을 때 발생할 수 있는 문제는 무엇인가요?

    - OneHotEncoder의 목적은 '서울', '부산'과 같이 순서가 없는 범주형 데이터를 머신러닝 모델이 인식할 수 있는 수치형 데이터로 변환하는 것입니다.

    - 범주형 데이터를 [1, 2, 3]과 같은 숫자로 그대로 변환하면, 모델은 이 숫자들 사이에 순서나 크기 관계가 있다고 오해할 수 있습니다. 
      예를 들어 '3'이 '1'보다 2배 더 중요하다거나, '2'가 '1'과 '3'의 중간값이라고 잘못 학습할 수 있습니다. 
      OneHotEncoder는 이러한 오해를 방지합니다.

Q2. OneHotEncoder의 drop='first' 매개변수는 어떤 역할을 하며, 이를 사용하면 어떤 장점이 있나요?

    - drop='first'는 각 범주형 특징에서 첫 번째 범주에 해당하는 컬럼을 제거하는 역할을 합니다.

    - 이 옵션을 사용하면 다중공선성(multicollinearity) 문제를 피할 수 있습니다. 
      예를 들어, City_Busan과 City_Jeju 컬럼이 모두 0이면, City_Seoul임을 알 수 있습니다. 
      즉, 하나의 범주가 다른 범주들의 조합으로 표현될 수 있기 때문에 
      불필요한 중복 정보를 제거하여 모델의 안정성을 높이는 장점이 있습니다.

      다중공선성을 제거하지 않으면 다음과 같은 문제가 발생할 수 있습니다:
      
        회귀 계수(Coefficients)의 불안정성:
            다중공선성이 있을 때, 회귀 모델은 각 변수의 중요도를 정확하게 측정하기 어렵게 됩니다.
            변수들의 상관관계가 높기 때문에, 특정 변수의 값이 조금만 바뀌어도 
            모델이 계산하는 회귀 계수(가중치)가 크게 변동할 수 있습니다. 
            이는 모델의 해석 가능성을 떨어뜨리고 예측 신뢰도를 낮춥니다.

        계수의 부호(Sign)가 잘못될 수 있음:
            변수가 가진 실제 효과와 반대되는 부호의 계수가 나올 수 있습니다. 
            예를 들어, 소득이 높을수록 구매액이 늘어나는 것이 상식적이지만, 
            다중공선성 때문에 소득 변수의 계수가 음수로 계산되는 기현상이 발생할 수 있습니다.

        통계적 유의성 판단의 어려움:
            각 독립 변수의 p-값(p-value)이 높게 나타나, 실제로는 중요한 변수임에도 불구하고 
            통계적으로 유의하지 않다고 잘못 판단할 수 있습니다.

Q3. ColumnTransformer를 사용하면 어떤 이점을 얻을 수 있으며, remainder='passthrough' 매개변수는 무엇을 의미하나요?

    - ColumnTransformer는 여러 컬럼에 각각 다른 전처리(예: 일부 컬럼에는 스케일러, 다른 컬럼에는 인코더)를 적용해야 할 때, 
      이 과정을 하나의 객체로 통합하여 관리할 수 있게 해줍니다. 
      이는 코드의 가독성을 높이고, Pipeline과 결합하여 전처리-모델링 과정을 자동화하는 데 매우 유용합니다.

    - remainder='passthrough'는 transformers에 지정되지 않은 나머지 컬럼들을 변환 없이 그대로 유지하라는 의미입니다. 
      데모 코드에서는 'Temperature' 컬럼이 이에 해당합니다.

Q4. 데모 코드에서 OneHotEncoder를 적용한 결과, 'City' 컬럼의 'Busan'은 어디로 갔으며, 'City_Seoul'과 'City_Jeju'만 남은 이유를 설명해 주세요.

    - OneHotEncoder의 drop='first' 매개변수 때문에 'City' 컬럼의 정렬된 값 중 첫 번째 범주인 'Busan'을 제외합니다. 
      get_feature_names_out을 통해 컬럼명을 얻으면 'City_Busan', 'City_Jeju', 'City_Seoul'이 되지만, 
      drop='first' 설정으로 인해 첫 번째 범주인 'City_Busan'이 제거됩니다.

      OneHotEncoder의 기본 정렬 순서는 알파벳 순서입니다. 
      따라서 'Busan'이 'Jeju', 'Seoul'보다 먼저 정렬되어 첫 번째 범주로 간주됩니다. 
      그래서 'City_Jeju'와 'City_Seoul'만 남습니다. 'Busan'은 두 컬럼이 모두 0일 때를 의미합니다.
      
      OneHotEncoder는 학습 데이터의 고유값을 찾아 정렬하는데, 이 때 알파벳 순서로 정렬합니다. 
      데모 데이터의 고유값은 'Busan', 'Jeju', 'Seoul'이며, 이 순서로 인코딩됩니다. 
      drop='first'는 첫 번째 컬럼인 'City_Busan'을 제거하므로, 'City_Jeju', 'City_Seoul' 컬럼만 남게 됩니다. 
      'Busan'은 두 컬럼이 모두 0일 때를 의미합니다.
"""
