"""
하이퍼파라미터 튜닝 (Hyperparameter Tuning)
모델의 성능을 극대화하기 위해 최적의 하이퍼파라미터 조합을 찾는 과정입니다. 단순히 한두 번 모델을 돌려보는 것을 넘어, 가장 좋은 성능을 내는 모델을 찾는 과학적인 방법입니다.
    sklearn.model_selection.GridSearchCV: 지정된 매개변수 값들을 하나씩 모두 테스트하여 가장 좋은 성능을 보이는 매개변수 조합을 찾아줍니다.
    sklearn.model_selection.RandomizedSearchCV: 모든 조합을 테스트하는 대신, 무작위로 매개변수 조합을 선택하여 테스트함으로써 효율성을 높입니다.

GridSearchCV는 모델의 하이퍼파라미터 튜닝을 위한 가장 대표적인 방법입니다.
사용자가 지정한 하이퍼파라미터 값들의 조합(그리드)을 모두 탐색하여,
교차 검증(Cross-Validation)을 통해 가장 높은 성능을 보이는 최적의 조합을 찾아줍니다.
"""

# 1. 필요한 라이브러리 임포트
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC  # 예시 모델: SVC (Support Vector Classifier)

# 2. 예제 데이터 생성
# 분류 문제 해결을 위한 가상 데이터셋
X, y = make_classification(n_samples=500, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 인스턴스 생성
# 하이퍼파라미터 튜닝의 대상이 될 기본 모델을 정의합니다.
svc = SVC(random_state=42)

# 4. 탐색할 하이퍼파라미터 그리드 정의
# 튜닝할 하이퍼파라미터들을 딕셔너리 형태로 정의합니다.
# 'C': 규제(regularization) 매개변수로, 값이 작을수록 규제가 강해집니다.
# 'gamma': 커널 함수의 계수로, 값이 클수록 훈련 데이터에 더 민감해집니다.
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel": ["rbf"],  # 이 예제에서는 'rbf' 커널만 사용
}
print("탐색할 하이퍼파라미터 그리드:")
print(param_grid)
print("-" * 50)

# 5. GridSearchCV 인스턴스 생성
# estimator: 튜닝할 기본 모델
# param_grid: 탐색할 하이퍼파라미터 그리드
# cv: 교차 검증 폴드 수
# scoring: 모델 성능을 평가할 지표 (여기서는 '정확도')
# verbose: 작업 진행 상황을 보여주는 상세도 (높을수록 상세)
grid_search = GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    verbose=2,
    n_jobs=-1,  # 모든 CPU 코어를 사용하여 병렬 처리 (학습 시간 단축)
)

# 6. GridSearchCV 학습 (하이퍼파라미터 탐색 시작)
# fit() 메서드를 호출하면 모든 하이퍼파라미터 조합에 대해 교차 검증을 수행합니다.
# 총 조합 수 (4 * 4 * 1) * 교차 검증 폴드 수 (5) = 80번의 학습이 진행됩니다.
print("GridSearchCV 하이퍼파라미터 탐색 시작...")
grid_search.fit(X_train, y_train)
print("GridSearchCV 하이퍼파라미터 탐색 완료!")
print("-" * 50)

# 7. 최적의 하이퍼파라미터 및 최고 성능 확인
# best_params_: 교차 검증을 통해 얻은 가장 좋은 성능을 낸 하이퍼파라미터 조합
print(f"최적의 하이퍼파라미터: {grid_search.best_params_}")
# best_score_: 최적의 하이퍼파라미터로 교차 검증했을 때의 평균 점수
print(f"최고 교차 검증 점수 (정확도): {grid_search.best_score_:.4f}")

# 8. 최종 모델 평가
# GridSearchCV는 최적의 하이퍼파라미터로 다시 학습된 모델을 내부적으로 저장합니다.
# 이 모델을 test 데이터셋에 적용하여 최종 성능을 평가합니다.
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"테스트 데이터셋에서의 최종 정확도: {final_accuracy:.4f}")
print("-" * 50)

# 참고: 모든 조합별 교차 검증 결과 확인
print("모든 매개변수 조합별 교차 검증 결과:")
print(grid_search.cv_results_)

"""
탐색할 하이퍼파라미터 그리드:
{'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
--------------------------------------------------
GridSearchCV 하이퍼파라미터 탐색 시작...
Fitting 5 folds for each of 16 candidates, totalling 80 fits
[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s
[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s
[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s
[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s
...
[CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s
GridSearchCV 하이퍼파라미터 탐색 완료!
--------------------------------------------------
최적의 하이퍼파라미터: {'C': 0.1, 'gamma': 0.01, 'kernel': 'rbf'}
최고 교차 검증 점수 (정확도): 0.8725
테스트 데이터셋에서의 최종 정확도: 0.9100
--------------------------------------------------
모든 매개변수 조합별 교차 검증 결과:
{'mean_fit_time': array([0.01839571, 0.01530485, 0.01561785, 0.01471353, 0.01637173,
       0.01195002, 0.00905228, 0.01321878, 0.01988659, 0.01715937,
       0.01059942, 0.01101189, 0.02129307, 0.01780772, 0.0171443 ,
       0.01000037]), 'std_fit_time': array([0.00223373, 0.00075003, 0.00083162, 0.00154257, 0.00117721,
       0.00074147, 0.0013472 , 0.00097681, 0.00108303, 0.00042722,
       0.00049109, 0.00131314, 0.0017248 , 0.00112461, 0.00154393,
       0.00063203]), 'mean_score_time': array([0.01110783, 0.01079998, 0.00982008, 0.00986176, 0.00759864,
       0.00570765, 0.00589366, 0.00951285, 0.01052833, 0.00945263,
       0.00539999, 0.00699944, 0.01048589, 0.00899606, 0.00540047,
       0.00580001]), 'std_score_time': array([0.00066784, 0.00097993, 0.00136087, 0.0007703 , 0.00119786,
       0.00039525, 0.00185585, 0.00045453, 0.00107804, 0.00055787,
       0.00048965, 0.00109489, 0.00117607, 0.00038442, 0.00048945,
       0.00040031]), 'param_C': masked_array(data=[0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0,
                   10.0, 10.0, 100.0, 100.0, 100.0, 100.0],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False],
       fill_value=1e+20), 'param_gamma': masked_array(data=[1.0, 0.1, 0.01, 0.001, 1.0, 0.1, 0.01, 0.001, 1.0, 0.1,
                   0.01, 0.001, 1.0, 0.1, 0.01, 0.001],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False],
       fill_value=1e+20), 'param_kernel': masked_array(data=['rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf',
                   'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False],
       fill_value=np.str_('?'),
            dtype=object), 'params': [{'C': 0.1, 'gamma': 1, 'kernel': 'rbf'}, {'C': 0.1, 'gamma': 0.1, 'kernel': 'rbf'}, {'C': 0.1, 'gamma': 0.01, 'kernel': 'rbf'}, {'C': 0.1, 'gamma': 0.001, 'kernel': 'rbf'}, {'C': 1, 'gamma': 1, 'kernel': 'rbf'}, {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}, {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}, {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}, {'C': 10, 'gamma': 1, 'kernel': 'rbf'}, {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}, {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}, {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}, {'C': 100, 'gamma': 1, 'kernel': 'rbf'}, {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'}, {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}, {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}], 'split0_test_score': array([0.5   , 0.4875, 0.8625, 0.5   , 0.5   , 0.8375, 0.85  , 0.8875,
       0.5   , 0.85  , 0.8625, 0.85  , 0.5   , 0.85  , 0.825 , 0.85  ]), 'split1_test_score': array([0.5125, 0.5625, 0.85  , 0.5125, 0.5125, 0.85  , 0.8125, 0.85  ,
       0.5125, 0.8375, 0.8125, 0.8375, 0.5125, 0.8375, 0.825 , 0.7625]), 'split2_test_score': array([0.5125, 0.525 , 0.875 , 0.5125, 0.5125, 0.875 , 0.875 , 0.8625,
       0.5125, 0.875 , 0.875 , 0.875 , 0.5125, 0.875 , 0.825 , 0.8875]), 'split3_test_score': array([0.5125, 0.5375, 0.9125, 0.5125, 0.5125, 0.8875, 0.875 , 0.9   ,
       0.5125, 0.875 , 0.8875, 0.875 , 0.5125, 0.875 , 0.85  , 0.875 ]), 'split4_test_score': array([0.5125, 0.525 , 0.8625, 0.5125, 0.5125, 0.8625, 0.9125, 0.8625,
       0.5125, 0.825 , 0.85  , 0.9   , 0.5125, 0.825 , 0.8625, 0.8625]), 'mean_test_score': array([0.51  , 0.5275, 0.8725, 0.51  
, 0.51  , 0.8625, 0.865 , 0.8725,
       0.51  , 0.8525, 0.8575, 0.8675, 0.51  , 0.8525, 0.8375, 0.8475]), 'std_test_score': array([0.005     , 0.0242384 , 0.02150581, 0.005     , 0.005     ,
       0.01767767, 0.03297726, 0.01837117, 0.005     , 0.02      ,
       0.02573908, 0.02179449, 0.005     , 0.02      , 0.01581139,
       0.04430011]), 'rank_test_score': array([12, 11,  1, 12, 12,  5,  4,  1, 12,  7,  6,  3, 12,  7, 10,  9],
      dtype=int32)}
"""

"""
복습

Q1. GridSearchCV는 어떤 방식으로 최적의 하이퍼파라미터를 찾나요? 이 방법의 장점과 단점은 무엇인가요?

    방식: 사용자가 지정한 하이퍼파라미터 값들의 **모든 가능한 조합(그리드)**을 하나씩 탐색합니다. 
         각 조합에 대해 cv로 지정된 횟수만큼 **교차 검증(Cross-Validation)**을 수행하고, 
         그 평균 점수가 가장 높은 조합을 최적의 하이퍼파라미터로 선택합니다.

    장점: 모든 조합을 검증하기 때문에 최적의 조합을 놓칠 위험이 없습니다.
    
    단점: 하이퍼파라미터의 개수나 탐색할 값의 범위가 넓을 경우, 
         학습에 필요한 시간이 기하급수적으로 증가하여 매우 비효율적일 수 있습니다.

Q2. GridSearchCV에서 fit() 메서드를 호출할 때, 내부적으로 어떤 과정이 진행되는지 설명해 주세요.
    
    fit()이 호출되면, GridSearchCV는 다음 과정을 수행합니다.

        1. param_grid에 정의된 모든 하이퍼파라미터 조합을 만듭니다.
        2. 각 조합에 대해 cv에 지정된 폴드 수만큼 데이터를 분할하여 교차 검증을 시작합니다.
        3. 교차 검증의 각 폴드에서, 학습 데이터로 모델을 학습시키고 테스트 데이터로 성능을 평가합니다.
        4. 모든 폴드에 대한 평가 점수를 평균하여 해당 조합의 최종 점수를 산출합니다.
        5. 모든 하이퍼파라미터 조합에 대해 이 과정을 반복합니다.
        6. 최종적으로 가장 높은 평균 점수를 낸 조합을 best_params_로 저장하고, 해당 조합으로 전체 학습 데이터에 다시 학습시킨 최종 모델을 best_estimator_에 저장합니다.

Q3. 코드에서 param_grid 딕셔너리에 'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]을 정의했고 cv=5로 설정했습니다. 
    이 경우 총 몇 번의 모델 학습이 이루어지나요?
    
    총 학습 횟수는 (C 값의 개수) * (gamma 값의 개수) * (kernel 값의 개수) * (교차 검증 폴드 수)입니다.
    4 * 4 * 1 * 5 = 80
    따라서 총 80번의 모델 학습이 이루어집니다.

Q4. GridSearchCV의 best_score_와 final_accuracy는 어떤 차이가 있나요? 왜 이 두 값을 모두 확인하는 것이 중요한가요?
    
    best_score_: 
        GridSearchCV가 학습 데이터셋 내에서 교차 검증을 통해 얻은 최고 평균 점수입니다. 
        이 값은 하이퍼파라미터 튜닝 과정에서 모델의 성능을 평가하기 위해 사용됩니다.
    
    final_accuracy: 
        GridSearchCV가 찾은 최적의 하이퍼파라미터로 학습된 최종 모델을 
        한 번도 보지 않은 테스트 데이터셋에 적용하여 얻은 점수입니다.
    
    이 두 값을 모두 확인하는 것이 중요한 이유는 과적합(Overfitting) 여부를 판단할 수 있기 때문입니다. 
    best_score_는 높지만 final_accuracy가 상대적으로 낮다면, 
    모델이 학습 데이터에 과적합되었을 가능성이 높습니다. 
    final_accuracy는 모델의 진정한 일반화 성능을 보여주는 지표입니다.
"""
