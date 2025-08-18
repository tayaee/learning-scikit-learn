"""
지도 학습 (Supervised Learning):
    분류 (Classification): 데이터를 미리 정의된 클래스 또는 범주로 분류하는 데 사용됩니다.
        예: sklearn.svm.SVC, sklearn.ensemble.RandomForestClassifier
    회귀 (Regression): 연속적인 숫자 값을 예측하는 데 사용됩니다.
        예: sklearn.linear_model.LinearRegression, sklearn.ensemble.RandomForestRegressor

RandomForestRegressor는 여러 개의 결정 트리(Decision Tree)를 만들고,
이 트리들의 예측을 종합해 최종 결과를 내는 앙상블(Ensemble) 모델입니다.
분류 문제에 사용되는 RandomForestClassifier와 매우 유사하지만, 연속적인 값을 예측하는 회귀 문제에 특화되어 있습니다.

RandomForestRegressor의 핵심 원리:
1.  **배깅 (Bagging, Bootstrap Aggregating)**:
    학습 데이터에서 무작위로 중복을 허용하여 여러 개의 서브셋(Bootstrap Sample)을 만듭니다.
    각 서브셋은 독립적인 결정 트리를 학습하는 데 사용됩니다.

2.  **무작위 특징 선택 (Random Feature Selection)**:
    각 트리의 노드를 분할할 때, 전체 특징 중에서 일부 특징만 무작위로 선택하여 최적의 분할을 찾습니다.
    이를 통해 각 트리가 서로 다른 특징을 기반으로 학습하게 되어, 트리 간의 상관관계를 줄이고 모델의 일반화 성능을 높입니다.

3.  **예측 (Averaging)**:
    새로운 데이터에 대한 예측을 수행할 때, 포레스트에 있는 모든 결정 트리가 각각의 예측값을 내놓습니다.
    RandomForestRegressor는 이 모든 예측값의 **평균**을 계산하여 최종 예측 결과로 사용합니다.
    (참고: RandomForestClassifier는 다수결 투표(Voting)를 사용합니다.)

이러한 과정을 통해 단일 결정 트리가 가질 수 있는 과적합(Overfitting) 문제를 효과적으로 해결하고, 더 안정적이고 높은 성능을 보입니다.
이 데모 코드를 통해 RandomForestRegressor의 학습 과정을 단계별로 살펴보겠습니다.
"""

# 1. 필요한 라이브러리 임포트
from sklearn.datasets import make_regression  # 회귀용 가상 데이터 생성
from sklearn.ensemble import RandomForestRegressor  # 랜덤 포레스트 회귀 모델
from sklearn.metrics import mean_squared_error, r2_score  # 모델 평가 지표
from sklearn.model_selection import train_test_split  # 데이터 분할

# 2. 예제 데이터 생성
# '중고차 가격'을 예측하는 비즈니스 시나리오를 모사합니다.
# - n_samples=200: 200개의 중고차 매물 데이터
# - n_features=10: 10개의 특징 (예: 연식, 주행거리, 엔진 크기, 브랜드, 사고 이력 등)
# - y (타겟 변수): '중고차 가격'
# - noise=20: 가격에 영향을 주는 예측 불가능한 요인 (예: 판매자의 급한 사정, 협상 능력 등)
# - random_state=42: 재현 가능한 결과를 위한 시드값
X, y = make_regression(n_samples=200, n_features=10, noise=20, random_state=42)

# 생성된 데이터의 범위 확인
# make_regression은 기본적으로 표준 정규 분포(평균 0, 표준편차 1)에서 값을 샘플링하므로,
# 값의 범위가 -1에서 1 사이로 제한되지 않습니다.
print(f"생성된 특징(X) 데이터의 최솟값: {X.min():.2f}")
print(f"생성된 특징(X) 데이터의 최댓값: {X.max():.2f}")
print(f"생성된 특징(X) 데이터의 표준편차: {X.std():.2f}")
print(f"생성된 특징(X) 데이터의 평균: {X.mean():.2f}")
print("-" * 50)

# 3. 데이터 분할
# 전체 데이터를 학습(train) 세트와 테스트(test) 세트로 나눕니다.
#
# [학습/테스트 데이터 분할 비율에 대한 고찰]
# 80:20, 70:30, 75:25 등 다양한 비율이 사용되며, 정해진 정답은 없습니다.
# 선택은 주로 전체 데이터셋의 크기에 따라 달라집니다.
# - 데이터셋이 클 경우 (수십만 개 이상): 90:10 또는 99:1과 같이 테스트 세트의 비율을 작게 가져가도,
#   테스트 샘플 수가 충분하여 안정적인 평가가 가능합니다.
# - 데이터셋이 작을 경우 (수백~수만 개): 80:20 또는 70:30 비율이 일반적으로 사용됩니다.
#   학습 데이터를 충분히 확보하면서도, 평가의 신뢰도를 유지하기 위한 균형점입니다.
#
# 이 예제에서는 80:20 비율을 사용하며, 이는 가장 널리 사용되는 표준적인 비율 중 하나입니다.
# 더 신뢰도 높은 검증을 위해서는 교차 검증(Cross-Validation)을 사용하는 것이 좋습니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 추정기(Estimator) 인스턴스 생성
# RandomForestRegressor 모델을 생성합니다.
#
# [n_estimators 값 선택에 대한 고찰]
# n_estimators는 포레스트를 구성하는 결정 트리의 개수를 의미하며, 모델 성능에 중요한 영향을 미칩니다.
#
# - 많을수록 좋은가?
#   일반적으로 트리의 개수가 많을수록 모델의 성능은 안정되고 예측의 분산이 줄어듭니다.
#   하지만 일정 수준을 넘어서면 성능 향상은 미미해지고, 학습 시간과 메모리 사용량만 증가하는 '수확 체감(Diminishing Returns)' 현상이 나타납니다.
#
# - Best Practice는?
#   데이터셋의 크기나 복잡도에 따라 최적의 값이 다르기 때문에 "데이터 개수 대비 몇 개"라는 절대적인 규칙은 없습니다.
#   대신, 다음과 같은 경험적인 방법(Heuristic)이 널리 사용됩니다.
#   1. 시작점: 일반적으로 100~300 사이의 값으로 시작하는 것이 좋은 출발점입니다. (scikit-learn의 기본값은 100입니다.)
#   2. 성능 곡선 확인: 트리의 개수를 늘려가면서 교차 검증(Cross-Validation) 점수나 OOB(Out-of-Bag) 점수의 변화를 시각화합니다.
#      성능이 더 이상 크게 향상되지 않고 안정되는 지점(Elbow point)을 찾는 것이 가장 이상적인 방법입니다.
#   3. 계산 비용 고려: 성능 향상이 거의 없는 지점에서는 굳이 더 많은 트리를 사용할 필요가 없습니다.
#      성능과 계산 비용 사이의 균형을 맞추는 것이 중요합니다.
#
# 결론: 100은 합리적인 기본값이지만, 최적의 성능을 위해서는 교차 검증이나 GridSearchCV와 같은 방법을 통해
#       자신의 데이터셋에 맞는 최적의 n_estimators 값을 찾는 것이 가장 좋습니다.
model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)

# 5. 모델 학습 (fit)
# model.fit(X, y) 메서드를 사용하여 학습 데이터에 모델을 맞춥니다.
# 이 과정을 통해 모델은 여러 개의 결정 트리를 생성하고 학습합니다.
print("모델 학습 시작...")
model.fit(X_train, y_train)
print("모델 학습 완료!")

# 6. 예측 (predict)
# model.predict(X) 메서드를 사용하여 학습된 모델로 테스트 데이터에 대한 예측을 수행합니다.
# 예측 결과는 연속적인 숫자 값(회귀 값)이 됩니다.
y_pred = model.predict(X_test)

# 7. 모델 평가
# 실제 값(y_test)과 예측 값(y_pred)을 비교하여 모델의 성능을 평가합니다.

# 평균 제곱 오차 (Mean Squared Error, MSE): 예측 값과 실제 값의 차이(오차)를 제곱한 값들의 평균
mse = mean_squared_error(y_test, y_pred)
print(f"\n평균 제곱 오차 (MSE): {mse:.2f}")

# 결정 계수 (R-squared, R2 Score): 모델이 분산을 얼마나 잘 설명하는지 나타내는 지표. 1에 가까울수록 좋습니다.
r2 = r2_score(y_test, y_pred)
print(f"결정 계수 (R2 Score): {r2:.2f}")

"""
실행 예

생성된 특징(X) 데이터의 최솟값: -2.64
생성된 특징(X) 데이터의 최댓값: 2.93
--------------------------------------------------
모델 학습 시작...
모델 학습 완료!

평균 제곱 오차 (MSE): 9167.42
결정 계수 (R2 Score): 0.74
"""

"""
복습

Q1. RandomForestRegressor는 RandomForestClassifier와 어떤 점에서 동일하고, 어떤 점에서 다른가요?

    - 동일한 점: 둘 다 여러 개의 결정 트리(Decision Tree)를 만들고 
      이 트리들의 예측을 종합하는 앙상블 학습(Ensemble Learning) 기법을 사용한다는 점에서 동일합니다. 
      n_estimators와 같은 주요 하이퍼파라미터도 공유합니다.

    - 다른 점: RandomForestClassifier는 분류를 위해 트리의 예측을 **다수결 투표(Voting)**로 종합하여 최종 클래스 레이블을 결정하는 반면, 
      RandomForestRegressor는 회귀를 위해 트리의 예측을 단순 평균(Averaging) 내어 최종 연속적인 값을 결정합니다.

Q2. 회귀 문제에서 RandomForestRegressor가 예측하는 최종 값은 어떻게 결정되나요?

    - RandomForestRegressor는 각 결정 트리가 예측한 값들을 모아 단순 평균을 내어 최종 예측 값으로 사용합니다. 
      예를 들어, 100개의 트리가 각각 예측한 값이 있다면, 그 100개의 값의 평균이 최종 예측 결과가 됩니다.

Q3. n_estimators 매개변수 외에 RandomForestRegressor의 성능에 영향을 줄 수 있는 다른 중요한 매개변수는 무엇이 있을까요?

    - max_depth: 각 결정 트리의 최대 깊이를 제한합니다. 
      이 값을 너무 깊게 설정하면 과적합(Overfitting)이 발생할 수 있습니다.
      기본값은 None으로 트리가 완전히 성장할 때까지 분할합니다.

    - min_samples_split: 노드를 분할하기 위한 최소 샘플 수를 지정합니다. 
      이 값을 높게 설정하면 모델이 간단해지고 과적합을 방지할 수 있습니다.

    - max_features: 각 결정 트리가 분할할 때 고려하는 특징(feature)의 최대 개수를 지정합니다. 
      이 값을 조정하여 트리의 다양성을 높일 수 있습니다.

Q4. mean_squared_error와 r2_score 중 어떤 지표가 모델의 성능을 더 직관적으로 이해하는 데 도움이 되나요? 그 이유는 무엇인가요?

    - **r2_score**가 일반적으로 모델의 성능을 더 직관적으로 이해하는 데 도움이 됩니다.

    - **r2_score**는 0에서 1 사이의 값으로 표현되며, 
      1에 가까울수록 모델이 데이터를 잘 설명한다는 의미를 가집니다. 
      이는 마치 '모델이 데이터의 변동성을 %로 얼마나 설명하는가'와 유사한 의미로 해석할 수 있어 이해하기 쉽습니다.

    - 반면, **mean_squared_error**는 오차의 제곱 평균이므로 그 자체의 단위가 없어 다른 데이터셋과 직접적으로 비교하기 어렵습니다. 
      값이 낮을수록 좋다는 것은 알 수 있지만, 20이라는 MSE 값이 좋은 것인지 나쁜 것인지는 기준이 없으면 판단하기 어렵습니다.
"""
