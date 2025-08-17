"""
지도 학습 (Supervised Learning):
    분류 (Classification): 데이터를 미리 정의된 클래스 또는 범주로 분류하는 데 사용됩니다.
        예: sklearn.svm.SVC, sklearn.ensemble.RandomForestClassifier

RandomForestClassifier:
    RandomForestClassifier는 앙상블 학습(Ensemble Learning) 기법 중 하나인 랜덤 포레스트를 사용한 분류 모델입니다.
    여러 개의 결정 트리(Decision Tree)를 만들고, 각 트리의 예측 결과를 다수결로 종합하여 최종 결과를 도출합니다.
    이렇게 여러 개의 트리를 사용하기 때문에 단일 트리가 가질 수 있는 과적합(Overfitting) 문제를 효과적으로 해결하고,
    더 안정적이고 높은 성능을 보입니다.
"""

# 1. 필요한 라이브러리 임포트
from sklearn.datasets import make_classification  # 분류용 가상 데이터 생성
from sklearn.ensemble import RandomForestClassifier  # 랜덤 포레스트 분류 모델
from sklearn.metrics import accuracy_score, classification_report  # 모델 평가 지표
from sklearn.model_selection import train_test_split  # 데이터 분할

# 2. 예제 데이터 생성
# 실제 데이터셋을 모사하기 위해 make_classification 함수를 사용하여 샘플 데이터를 생성합니다.
# 이 데이터는 1000개의 샘플(n_samples)을 가지며, 각 샘플은 20개의 특징(n_features)을 가집니다.
# 이 중 10개의 특징(n_informative)은 클래스 분류에 유의미한 정보를 포함하고 있습니다.
# 데이터는 2개의 클래스(n_classes)로 나뉘며, 재현 가능한 결과를 위해 random_state를 42로 설정합니다.
# 예를 들어, 이 데이터는 고객의 구매 이력(특징)을 바탕으로 구매 여부(클래스)를 예측하는 시나리오를 모사할 수 있습니다.
#
# 그외에도 make_regression, make_blobs, make_moons, make_circles, make_friedman1 등 다양한 데이터 생성 함수가 있습니다.
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# 3. 데이터 분할
# 전체 데이터를 학습(train) 세트와 테스트(test) 세트로 나눕니다.
# 학습 세트(80%)는 모델 훈련에 사용하고, 테스트 세트(20%)는 모델 성능 평가에 사용합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 추정기(Estimator) 인스턴스 생성
# RandomForestClassifier 모델을 생성합니다.
# n_estimators: 포레스트에 포함될 결정 트리의 개수. 많을수록 일반적으로 성능이 좋아지지만, 학습 시간이 길어집니다.
# random_state: 모델의 결과를 고정하여 코드 재실행 시에도 동일한 결과를 얻도록 합니다.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5. 모델 학습 (fit)
# model.fit(X, y) 메서드를 사용하여 학습 데이터에 모델을 맞춥니다.
# 이 과정을 통해 모델은 여러 개의 결정 트리를 생성하고 학습합니다.
print("모델 학습 시작...")
model.fit(X_train, y_train)
print("모델 학습 완료!")

# 6. 예측 (predict)
# model.predict(X) 메서드를 사용하여 학습된 모델로 테스트 데이터에 대한 예측을 수행합니다.
# 예측 결과는 0 또는 1과 같은 클래스 레이블이 됩니다.
y_pred = model.predict(X_test)

# 7. 모델 평가
# 실제 값(y_test)과 예측 값(y_pred)을 비교하여 모델의 성능을 평가합니다.

# 정확도 (Accuracy): 전체 예측 중 올바르게 예측한 비율을 계산합니다.
accuracy = accuracy_score(y_test, y_pred)
print(f"\n모델의 accuracy: {accuracy:.2f}")

# Classification Report: 정밀도(Precision), 재현율(Recall), F1-점수 등 더 자세한 평가 지표를 제공합니다.
# precision: '양성'으로 예측한 것 중에서 실제 '양성'인 비율
# recall: 실제 '양성' 중에서 모델이 '양성'으로 올바르게 예측한 비율
# f1-score: precision과 recall의 조화 평균으로, 두 지표의 균형을 평가합니다.
# support: 각 클래스의 샘플 수
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

"""
모델 학습 시작...
모델 학습 완료!

모델의 accuracy: 0.92   

Classification Report:
              precision    recall  f1-score   support
                    [2]       [3]       [4]       [5]

           0       0.95      0.89      0.92       104
           1       0.89      0.95      0.92        96

    accuracy                           0.92       200 [1]
   macro avg       0.92      0.92      0.92       200 [6]
weighted avg       0.92      0.92      0.92       200 [7]

Report 해석 방법:

[1] "accuracy 0.92 200"는 전체 200개의 테스트 샘플 중 92%가 올바르게 분류되었음을 의미.
    즉 precision 아래의 0.95와 0.89를 종합한 결과로, 모델이 전체적으로 92%의 정확도를 보였다는 것을 나타냅니다.

클래스별(0, 1) 상세 지표:

[2] precision: 클래스 0에 대해 0.95, 클래스 1에 대해 0.89로, 
    클래스 0을 '양성'으로 예측한 것 중 실제로 '양성'인 비율이 95%, 클래스 1은 89%임을 의미.

[3] recall: 클래스 0에 대해 0.89, 클래스 1에 대해 0.95로, 
    실제 '양성'인 샘플 중 모델이 올바르게 예측한 비율이 클래스 0은 89%, 클래스 1은 95%임을 의미.

[4] f1-score: precision과 recall의 조화 평균으로, 클래스 0은 0.92, 클래스 1도 0.92로, 
    두 클래스 모두 균형 잡힌 성능을 보임을 나타냅니다. 둘 중 하나가 낮으면 f1 점수가 낮게 나옵니다. 
    둘 다 조화롭게 높은 경우 f1 점수도 높게 나옵니다.

[5] support: 각 클래스의 실제 샘플 수로, 클래스 0은 104개, 클래스 1은 96개로 나타납니다. 
    이는 데이터셋에서 각 클래스가 얼마나 균형 잡혀 있는지를 보여줍니다.

[6] macro avg: support를 제외한 나머지 클래스 지표(precision, recall, f1-score)의 단순 평균으로, 각 클래스의 성능을 균등하게 고려합니다.

[7] weighted avg: 각 클래스의 support를 고려한 가중 평균으로, 클래스의 샘플 수에 따라 지표가 조정됩니다.
    weighted_precision = (precision_0 * support_0 + precision_1 * support_1) / (support_0 + support_1)
    weighted_recall = (recall_0 * support_0 + recall_1 * support_1) / (support_0 + support_1)
    weighted_f1-score = (f1-score_0 * support_0 + f1-score_1 * support_1) / (support_0 + support_1)

전체적으로 해석 방식을 다시 요약:

1. 전체 성능 평가: accuracy르 ㄹ통해 전반적인 예측 능력을 파악합니다.

2. 클래스 불균형 확인: support를 통해 각 클래스의 샘플 수를 확인하여 데이터셋의 균형 상태를 이해합니다. 이 예에서는 104, 96으로 비교적 균형 잡힌 상태입니다.

3. 클래스별 성능 상세 분석: precision, recall, f1-score를 통해 각 클래스별 성능을 세부적으로 파악하여 모델의 강점과 약점을 파악합니다.

class 0 의 경우 0.95 > 0.89 이므로, "클래스 0이라고 말하는 것에 대해' 매우 신중하고 정확하다는 의미입니다.
class 1 의 경우 0.89 < 0.95 이므로, "클래스 1인 것을 놓치지 않고' 잘 찾아낸다는 의미입니다.

4. 균형 평가: f1-score를 보니 둘 다 0.92로 매우 높은 수치를 보여주므로 전반적으로 균형 잡힌 성능을 보여줍니다.

"""

"""
복습

Q1. RandomForestClassifier는 어떤 원리로 분류를 수행하나요?

    - RandomForestClassifier는 앙상블 학습(Ensemble Learning) 기법을 사용합니다. 
      여러 개의 독립적인 결정 트리를 만들고, 각 트리가 내린 예측을 모아 다수결 투표 방식으로 최종 분류 결과를 결정합니다. 
      이를 통해 단일 결정 트리의 단점인 과적합(Overfitting)을 줄이고 모델의 안정성과 정확도를 높일 수 있습니다.

Q2. n_estimators 매개변수의 역할은 무엇인가요? 이 값을 늘리면 어떤 장점과 단점이 있을까요?

    - **n_estimators**는 RandomForestClassifier 모델을 구성하는 결정 트리의 개수를 지정하는 매개변수입니다.

    - 장점: n_estimators를 늘리면 더 많은 트리가 예측에 참여하여 모델의 정확도가 높아지고 일반화 성능이 향상됩니다.

    - 단점: 트리가 많아질수록 모델 학습 시간이 길어지고 메모리 사용량이 증가합니다.

Q3. 코드에서 fit()과 predict() 메서드의 역할과 사용 순서를 설명해 주세요.

    - fit() 메서드는 X_train과 y_train 데이터를 이용해 모델을 학습시킵니다. 
      이 과정에서 모델은 데이터의 패턴을 파악하고 최적의 모델 파라미터를 찾습니다.

    - predict() 메서드는 학습이 완료된 모델에 새로운 데이터(X_test)를 입력해 예측을 수행합니다.

    - 이 두 메서드는 fit() -> predict() 순서로 사용해야 합니다. 
      모델을 먼저 학습시키지 않으면 예측을 할 수 없기 때문입니다.

Q4. accuracy_score 외에 classification_report를 사용해 모델을 평가하는 것이 왜 유용한가요?

    - **accuracy_score**는 전체 예측 중 정확하게 맞힌 비율만 알려주기 때문에, 
      데이터의 클래스가 불균형할 경우(예: 한 클래스 데이터가 다른 클래스보다 훨씬 많을 경우) 모델 성능을 오해할 수 있습니다.

    - **classification_report**는 정밀도(Precision), 재현율(Recall), F1-점수 등 
      각 클래스별로 세분화된 성능 지표를 제공합니다. 
      이를 통해 모델이 특정 클래스를 얼마나 잘 예측하는지, 어떤 클래스 예측에 어려움을 겪는지 등 더 깊이 있는 분석이 가능합니다.
"""
