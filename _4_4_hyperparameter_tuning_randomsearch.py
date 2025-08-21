"""
하이퍼파라미터 튜닝 (Hyperparameter Tuning)
모델의 성능을 극대화하기 위해 최적의 하이퍼파라미터 조합을 찾는 과정입니다. 단순히 한두 번 모델을 돌려보는 것을 넘어, 가장 좋은 성능을 내는 모델을 찾는 과학적인 방법입니다.
    sklearn.model_selection.GridSearchCV: 지정된 매개변수 값들을 하나씩 모두 테스트하여 가장 좋은 성능을 보이는 매개변수 조합을 찾아줍니다.
    sklearn.model_selection.RandomizedSearchCV: 모든 조합을 테스트하는 대신, 무작위로 매개변수 조합을 선택하여 테스트함으로써 효율성을 높입니다.

RandomizedSearchCV는 GridSearchCV와 마찬가지로 하이퍼파라미터 튜닝을 위한 도구입니다.
하지만 모든 매개변수 조합을 탐색하는 GridSearchCV와 달리,
사용자가 정의한 분포(distribution)를 기반으로 일정 횟수만큼 무작위로 매개변수 조합을 샘플링하여 탐색합니다.
이 방식은 탐색 공간이 매우 넓을 때 GridSearchCV보다 훨씬 효율적으로 최적의 하이퍼파라미터를 찾을 수 있다는 장점이 있습니다.
"""

# 1. 필요한 라이브러리 임포트
import json
import os
from scipy.stats import uniform  # 균일 분포를 위한 라이브러리
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC

# 2. 예제 데이터 생성
# 분류 문제 해결을 위한 가상 데이터셋
X, y = make_classification(n_samples=500, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 인스턴스 생성
# 하이퍼파라미터 튜닝의 대상이 될 기본 모델을 정의합니다.
svc = SVC(random_state=42)

params = svc.get_params()
print(f"Model params: {json.dumps(params, indent=2)}")


# 4. 탐색할 하이퍼파라미터 분포 정의
# 'GridSearchCV'와 달리 값의 리스트 대신 분포를 정의할 수 있습니다.
# uniform(loc, scale): loc부터 loc+scale까지 균일 분포를 생성합니다.
param_distributions = {
    "C": uniform(loc=0.1, scale=100),  # 0.1부터 100.1까지 균일 분포에서 무작위 샘플링
    "gamma": uniform(loc=0.0001, scale=1),  # 0.0001부터 1.0001까지 균일 분포에서 무작위 샘플링
    "kernel": ["rbf"],
}
print("탐색할 하이퍼파라미터 분포:")
print(param_distributions)
print("-" * 50)

# 5. RandomizedSearchCV 인스턴스 생성
# estimator: 튜닝할 기본 모델
# param_distributions: 탐색할 하이퍼파라미터 분포
# n_iter: 무작위로 샘플링하여 탐색할 횟수
# cv: 교차 검증 폴드 수
# scoring: 모델 성능을 평가할 지표
# verbose: 작업 진행 상황을 보여주는 상세도
print(f"os.cpu_count() = {os.cpu_count()}")
random_search = RandomizedSearchCV(
    estimator=svc,
    param_distributions=param_distributions,
    n_iter=20,  # 20개의 무작위 조합만 탐색
    cv=5,
    scoring="accuracy",
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

# 6. RandomizedSearchCV 학습 (하이퍼파라미터 탐색 시작)
# n_iter=20이므로 총 20개의 무작위 조합에 대해 교차 검증을 수행합니다.
# 각 조합마다 5개의 폴드를 사용하므로 총 20 * 5 = 100번의 학습이 진행됩니다.
print("RandomizedSearchCV 하이퍼파라미터 탐색 시작...")
random_search.fit(X_train, y_train)
print("RandomizedSearchCV 하이퍼파라미터 탐색 완료!")
print("-" * 50)

# 7. 최적의 하이퍼파라미터 및 최고 성능 확인
# best_params_: 교차 검증을 통해 얻은 가장 좋은 성능을 낸 하이퍼파라미터 조합
print(f"최적의 하이퍼파라미터: {random_search.best_params_}")
# best_score_: 최적의 하이퍼파라미터로 교차 검증했을 때의 평균 점수
print(f"최고 교차 검증 점수 (정확도): {random_search.best_score_:.4f}")

# 8. 최종 모델 평가
# 최적의 하이퍼파라미터로 다시 학습된 모델을 test 데이터셋에 적용하여 최종 성능을 평가합니다.
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"테스트 데이터셋에서의 최종 정확도: {final_accuracy:.4f}")
print("-" * 50)

# 참고: 모든 조합별 교차 검증 결과 확인
print("모든 매개변수 조합별 교차 검증 결과:")
print(random_search.cv_results_)

"""
탐색할 하이퍼파라미터 분포:
{'C': <scipy.stats._distn_infrastructure.rv_continuous_frozen object at 0x000001CCF0F73C10>, 'gamma': <scipy.stats._distn_infrastructure.rv_continuous_frozen object at 0x000001CCF07C8FD0>, 'kernel': ['rbf']}
--------------------------------------------------
RandomizedSearchCV 하이퍼파라미터 탐색 시작...
Fitting 5 folds for each of 20 candidates, totalling 100 fits
[CV] END C=37.55401188473625, gamma=0.9508143064099162, kernel=rbf; total time=   0.0s
[CV] END C=37.55401188473625, gamma=0.9508143064099162, kernel=rbf; total time=   0.0s
[CV] END C=37.55401188473625, gamma=0.9508143064099162, kernel=rbf; total time=   0.0s
[CV] END C=83.34426408004217, gamma=0.21243911067827614, kernel=rbf; total time=   0.0s
[CV] END C=83.34426408004217, gamma=0.21243911067827614, kernel=rbf; total time=   0.0s
[CV] END C=83.34426408004217, gamma=0.21243911067827614, kernel=rbf; total time=   0.0s
[CV] END C=18.282496720710064, gamma=0.1835045098534338, kernel=rbf; total time=   0.0s
[CV] END C=18.282496720710064, gamma=0.1835045098534338, kernel=rbf; total time=   0.0s
[CV] END C=18.282496720710064, gamma=0.1835045098534338, kernel=rbf; total time=   0.0s
[CV] END C=18.282496720710064, gamma=0.1835045098534338, kernel=rbf; total time=   0.0s
[CV] END C=73.2993941811405, gamma=0.5987584841970366, kernel=rbf; total time=   0.0s
[CV] END C=18.282496720710064, gamma=0.1835045098534338, kernel=rbf; total time=   0.0s
[CV] END C=30.524224295953772, gamma=0.5248564316322378, kernel=rbf; total time=   0.0s
[CV] END C=73.2993941811405, gamma=0.5987584841970366, kernel=rbf; total time=   0.0s
[CV] END C=30.524224295953772, gamma=0.5248564316322378, kernel=rbf; total time=   0.0s
[CV] END C=30.524224295953772, gamma=0.5248564316322378, kernel=rbf; total time=   0.0s
[CV] END C=73.2993941811405, gamma=0.5987584841970366, kernel=rbf; total time=   0.0s
[CV] END C=30.524224295953772, gamma=0.5248564316322378, kernel=rbf; total time=   0.0s
[CV] END C=30.524224295953772, gamma=0.5248564316322378, kernel=rbf; total time=   0.0s
[CV] END C=43.294501864211576, gamma=0.2913291401980419, kernel=rbf; total time=   0.0s
[CV] END C=43.294501864211576, gamma=0.2913291401980419, kernel=rbf; total time=   0.0s
[CV] END C=43.294501864211576, gamma=0.2913291401980419, kernel=rbf; total time=   0.0s
[CV] END C=43.294501864211576, gamma=0.2913291401980419, kernel=rbf; total time=   0.0s
[CV] END C=43.294501864211576, gamma=0.2913291401980419, kernel=rbf; total time=   0.0s
[CV] END C=61.28528947223795, gamma=0.13959386065204182, kernel=rbf; total time=   0.0s
[CV] END C=61.28528947223795, gamma=0.13959386065204182, kernel=rbf; total time=   0.0s
[CV] END C=61.28528947223795, gamma=0.13959386065204182, kernel=rbf; total time=   0.0s
[CV] END C=61.28528947223795, gamma=0.13959386065204182, kernel=rbf; total time=   0.0s
[CV] END C=15.701864044243651, gamma=0.15609452033620264, kernel=rbf; total time=   0.0s
[CV] END C=61.28528947223795, gamma=0.13959386065204182, kernel=rbf; total time=   0.0s
[CV] END C=29.314464853521816, gamma=0.3664618432936917, kernel=rbf; total time=   0.0s
[CV] END C=29.314464853521816, gamma=0.3664618432936917, kernel=rbf; total time=   0.0s
[CV] END C=29.314464853521816, gamma=0.3664618432936917, kernel=rbf; total time=   0.0s
[CV] END C=29.314464853521816, gamma=0.3664618432936917, kernel=rbf; total time=   0.0s
[CV] END C=29.314464853521816, gamma=0.3664618432936917, kernel=rbf; total time=   0.0s
[CV] END C=45.706998421703595, gamma=0.7852759613930136, kernel=rbf; total time=   0.0s
[CV] END C=45.706998421703595, gamma=0.7852759613930136, kernel=rbf; total time=   0.0s
[CV] END C=45.706998421703595, gamma=0.7852759613930136, kernel=rbf; total time=   0.0s
[CV] END C=45.706998421703595, gamma=0.7852759613930136, kernel=rbf; total time=   0.0s
[CV] END C=45.706998421703595, gamma=0.7852759613930136, kernel=rbf; total time=   0.0s
[CV] END C=20.067378215835976, gamma=0.5143344384136116, kernel=rbf; total time=   0.0s
[CV] END C=20.067378215835976, gamma=0.5143344384136116, kernel=rbf; total time=   0.0s
[CV] END C=20.067378215835976, gamma=0.5143344384136116, kernel=rbf; total time=   0.0s
[CV] END C=20.067378215835976, gamma=0.5143344384136116, kernel=rbf; total time=   0.0s
[CV] END C=59.34145688620425, gamma=0.04655041271999773, kernel=rbf; total time=   0.0s
[CV] END C=20.067378215835976, gamma=0.5143344384136116, kernel=rbf; total time=   0.0s
[CV] END C=5.908361216819946, gamma=0.8662761457749352, kernel=rbf; total time=   0.0s
[CV] END C=59.34145688620425, gamma=0.04655041271999773, kernel=rbf; total time=   0.0s
[CV] END C=59.34145688620425, gamma=0.04655041271999773, kernel=rbf; total time=   0.0s
[CV] END C=59.34145688620425, gamma=0.04655041271999773, kernel=rbf; total time=   0.0s
[CV] END C=59.34145688620425, gamma=0.04655041271999773, kernel=rbf; total time=   0.0s
[CV] END C=60.85448519014384, gamma=0.17062412368729152, kernel=rbf; total time=   0.0s
[CV] END C=15.701864044243651, gamma=0.15609452033620264, kernel=rbf; total time=   0.0s
[CV] END C=5.908361216819946, gamma=0.8662761457749352, kernel=rbf; total time=   0.0s
[CV] END C=60.85448519014384, gamma=0.17062412368729152, kernel=rbf; total time=   0.0s
[CV] END C=60.85448519014384, gamma=0.17062412368729152, kernel=rbf; total time=   0.0s
[CV] END C=60.85448519014384, gamma=0.17062412368729152, kernel=rbf; total time=   0.0s
[CV] END C=60.85448519014384, gamma=0.17062412368729152, kernel=rbf; total time=   0.0s
[CV] END C=6.605159298527951, gamma=0.9489855372533332, kernel=rbf; total time=   0.0s
[CV] END C=6.605159298527951, gamma=0.9489855372533332, kernel=rbf; total time=   0.0s
[CV] END C=6.605159298527951, gamma=0.9489855372533332, kernel=rbf; total time=   0.0s
[CV] END C=6.605159298527951, gamma=0.9489855372533332, kernel=rbf; total time=   0.0s
[CV] END C=96.66320330745593, gamma=0.8084973481164611, kernel=rbf; total time=   0.0s
[CV] END C=6.605159298527951, gamma=0.9489855372533332, kernel=rbf; total time=   0.0s
[CV] END C=96.66320330745593, gamma=0.8084973481164611, kernel=rbf; total time=   0.0s
[CV] END C=5.908361216819946, gamma=0.8662761457749352, kernel=rbf; total time=   0.0s
[CV] END C=96.66320330745593, gamma=0.8084973481164611, kernel=rbf; total time=   0.0s
[CV] END C=96.66320330745593, gamma=0.8084973481164611, kernel=rbf; total time=   0.0s
[CV] END C=96.66320330745593, gamma=0.8084973481164611, kernel=rbf; total time=   0.0s
[CV] END C=30.56137691733707, gamma=0.09777211400638387, kernel=rbf; total time=   0.0s
[CV] END C=30.56137691733707, gamma=0.09777211400638387, kernel=rbf; total time=   0.0s
[CV] END C=30.56137691733707, gamma=0.09777211400638387, kernel=rbf; total time=   0.0s
[CV] END C=30.56137691733707, gamma=0.09777211400638387, kernel=rbf; total time=   0.0s
[CV] END C=30.56137691733707, gamma=0.09777211400638387, kernel=rbf; total time=   0.0s
[CV] END C=68.52330265121569, gamma=0.4402524937396013, kernel=rbf; total time=   0.0s
[CV] END C=68.52330265121569, gamma=0.4402524937396013, kernel=rbf; total time=   0.0s
[CV] END C=68.52330265121569, gamma=0.4402524937396013, kernel=rbf; total time=   0.0s
[CV] END C=37.55401188473625, gamma=0.9508143064099162, kernel=rbf; total time=   0.0s
[CV] END C=68.52330265121569, gamma=0.4402524937396013, kernel=rbf; total time=   0.0s
[CV] END C=68.52330265121569, gamma=0.4402524937396013, kernel=rbf; total time=   0.0s
[CV] END C=37.55401188473625, gamma=0.9508143064099162, kernel=rbf; total time=   0.0s
[CV] END C=60.21150117432088, gamma=0.7081725777960455, kernel=rbf; total time=   0.0s
[CV] END C=60.21150117432088, gamma=0.7081725777960455, kernel=rbf; total time=   0.0s
[CV] END C=73.2993941811405, gamma=0.5987584841970366, kernel=rbf; total time=   0.0s
[CV] END C=73.2993941811405, gamma=0.5987584841970366, kernel=rbf; total time=   0.0s
[CV] END C=15.701864044243651, gamma=0.15609452033620264, kernel=rbf; total time=   0.0s
[CV] END C=2.1584494295802448, gamma=0.9700098521619943, kernel=rbf; total time=   0.0s
[CV] END C=2.1584494295802448, gamma=0.9700098521619943, kernel=rbf; total time=   0.0s
[CV] END C=5.908361216819946, gamma=0.8662761457749352, kernel=rbf; total time=   0.0s
[CV] END C=15.701864044243651, gamma=0.15609452033620264, kernel=rbf; total time=   0.0s
[CV] END C=15.701864044243651, gamma=0.15609452033620264, kernel=rbf; total time=   0.0s
[CV] END C=60.21150117432088, gamma=0.7081725777960455, kernel=rbf; total time=   0.0s
[CV] END C=60.21150117432088, gamma=0.7081725777960455, kernel=rbf; total time=   0.0s
[CV] END C=60.21150117432088, gamma=0.7081725777960455, kernel=rbf; total time=   0.0s
[CV] END C=5.908361216819946, gamma=0.8662761457749352, kernel=rbf; total time=   0.0s
[CV] END C=2.1584494295802448, gamma=0.9700098521619943, kernel=rbf; total time=   0.0s
[CV] END C=83.34426408004217, gamma=0.21243911067827614, kernel=rbf; total time=   0.0s
[CV] END C=2.1584494295802448, gamma=0.9700098521619943, kernel=rbf; total time=   0.0s
[CV] END C=2.1584494295802448, gamma=0.9700098521619943, kernel=rbf; total time=   0.0s
[CV] END C=83.34426408004217, gamma=0.21243911067827614, kernel=rbf; total time=   0.0s
RandomizedSearchCV 하이퍼파라미터 탐색 완료!
--------------------------------------------------
최적의 하이퍼파라미터: {'C': np.float64(59.34145688620425), 'gamma': np.float64(0.04655041271999773), 'kernel': 'rbf'}
최고 교차 검증 점수 (정확도): 0.8575
테스트 데이터셋에서의 최종 정확도: 0.8600
--------------------------------------------------
모든 매개변수 조합별 교차 검증 결과:
{'mean_fit_time': array([0.02130742, 0.01929746, 0.01660695, 0.02062106, 0.01820703,
       0.01610756, 0.01594458, 0.01764688, 0.01822062, 0.01810708,
       0.01798582, 0.01831441, 0.02009768, 0.0185173 , 0.01587844,
       0.0169672 , 0.02022839, 0.01916642, 0.01709776, 0.01857595]), 'std_fit_time': array([0.00165557, 0.00312994, 0.0023723 , 0.00140278, 0.00221557,
       0.00246028, 0.00183302, 0.0007835 , 0.00114608, 0.00040581,
       0.00086469, 0.00042244, 0.00249609, 0.00093879, 0.00072694,
       0.00095171, 0.00182032, 0.00075235, 0.00049571, 0.00065806]), 'mean_score_time': array([0.01129899, 0.00952578, 0.00927539, 0.00981965, 0.00819988,
       0.00739861, 0.00866036, 0.01078496, 0.0105772 , 0.0101295 ,
       0.01005211, 0.0106389 , 0.01012163, 0.01016927, 0.00716147,
       0.01040177, 0.01112843, 0.01053896, 0.00946593, 0.01058865]), 'std_score_time': array([0.00109023, 0.00181579, 0.00125362, 0.00095619, 0.00074847,
       0.00101675, 0.00210195, 0.00067804, 0.00058698, 0.00126576,
       0.00046465, 0.00029145, 0.00046623, 0.00033546, 0.00048027,
       0.00049318, 0.00081494, 0.00038211, 0.00065887, 0.00048093]), 'param_C': masked_array(data=[37.55401188473625, 73.2993941811405,
                   15.701864044243651, 5.908361216819946,
                   60.21150117432088, 2.1584494295802448,
                   83.34426408004217, 18.282496720710064,
                   30.524224295953772, 43.294501864211576,
                   61.28528947223795, 29.314464853521816,
                   45.706998421703595, 20.067378215835976,
                   59.34145688620425, 60.85448519014384,
                   6.605159298527951, 96.66320330745593,
                   30.56137691733707, 68.52330265121569],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False],
       fill_value=1e+20), 'param_gamma': masked_array(data=[0.9508143064099162, 0.5987584841970366,
                   0.15609452033620264, 0.8662761457749352,
                   0.7081725777960455, 0.9700098521619943,
                   0.21243911067827614, 0.1835045098534338,
                   0.5248564316322378, 0.2913291401980419,
                   0.13959386065204182, 0.3664618432936917,
                   0.7852759613930136, 0.5143344384136116,
                   0.04655041271999773, 0.17062412368729152,
                   0.9489855372533332, 0.8084973481164611,
                   0.09777211400638387, 0.4402524937396013],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False],
       fill_value=1e+20), 'param_kernel': masked_array(data=['rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf',
                   'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf',
                   'rbf', 'rbf', 'rbf', 'rbf'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False],
       fill_value=np.str_('?'),
            dtype=object), 'params': [{'C': np.float64(37.55401188473625), 'gamma': np.float64(0.9508143064099162), 'kernel': 'rbf'}, {'C': np.float64(73.2993941811405), 'gamma': np.float64(0.5987584841970366), 'kernel': 'rbf'}, {'C': np.float64(15.701864044243651), 'gamma': np.float64(0.15609452033620264), 'kernel': 'rbf'}, {'C': np.float64(5.908361216819946), 'gamma': np.float64(0.8662761457749352), 'kernel': 'rbf'}, {'C': np.float64(60.21150117432088), 'gamma': np.float64(0.7081725777960455), 'kernel': 'rbf'}, {'C': np.float64(2.1584494295802448), 'gamma': np.float64(0.9700098521619943), 'kernel': 'rbf'}, {'C': np.float64(83.34426408004217), 'gamma': np.float64(0.21243911067827614), 'kernel': 'rbf'}, {'C': np.float64(18.282496720710064), 'gamma': np.float64(0.1835045098534338), 'kernel': 'rbf'}, {'C': np.float64(30.524224295953772), 'gamma': np.float64(0.5248564316322378), 'kernel': 'rbf'}, {'C': np.float64(43.294501864211576), 'gamma': np.float64(0.2913291401980419), 'kernel': 'rbf'}, {'C': np.float64(61.28528947223795), 'gamma': np.float64(0.13959386065204182), 'kernel': 'rbf'}, {'C': np.float64(29.314464853521816), 'gamma': 
np.float64(0.3664618432936917), 'kernel': 'rbf'}, {'C': np.float64(45.706998421703595), 'gamma': np.float64(0.7852759613930136), 'kernel': 'rbf'}, {'C': np.float64(20.067378215835976), 'gamma': np.float64(0.5143344384136116), 'kernel': 'rbf'}, {'C': np.float64(59.34145688620425), 'gamma': np.float64(0.04655041271999773), 'kernel': 'rbf'}, {'C': np.float64(60.85448519014384), 'gamma': np.float64(0.17062412368729152), 'kernel': 'rbf'}, {'C': np.float64(6.605159298527951), 'gamma': np.float64(0.9489855372533332), 'kernel': 'rbf'}, {'C': np.float64(96.66320330745593), 'gamma': np.float64(0.8084973481164611), 'kernel': 'rbf'}, {'C': np.float64(30.56137691733707), 'gamma': np.float64(0.09777211400638387), 'kernel': 'rbf'}, {'C': np.float64(68.52330265121569), 'gamma': np.float64(0.4402524937396013), 'kernel': 'rbf'}], 'split0_test_score': array([0.5   , 0.4875, 0.8125, 0.4875, 0.4875, 0.5   , 0.8375, 0.8375,
       0.4875, 0.7125, 0.8375, 0.575 , 0.4875, 0.4875, 0.875 , 0.825 ,
       0.5   , 0.4875, 0.8375, 0.525 ]), 'split1_test_score': array([0.5125, 0.5125, 0.875 , 0.5125, 0.5125, 0.5125, 0.85  , 0.875 ,
       0.5   , 0.8   , 0.875 , 0.6875, 0.5125, 0.5   , 0.85  , 0.875 ,
       0.5125, 0.5125, 0.8375, 0.5625]), 'split2_test_score': array([0.5125, 0.5125, 0.875 , 0.5125, 0.5125, 0.5125, 0.8375, 0.85  ,
       0.5125, 0.725 , 0.875 , 0.6   , 0.5125, 0.5125, 0.8625, 0.85  ,
       0.5125, 0.5125, 0.875 , 0.5375]), 'split3_test_score': array([0.5125, 0.5125, 0.8875, 0.5125, 0.5125, 0.5125, 0.8875, 0.9   ,
       0.5375, 0.7875, 0.8875, 0.65  , 0.5125, 0.55  , 0.85  , 0.9   ,
       0.5125, 0.5125, 0.875 , 0.5875]), 'split4_test_score': array([0.5125, 0.5125, 0.8   , 0.5125, 0.5125, 0.5125, 0.7625, 0.775 ,
       0.5125, 0.7125, 0.8125, 0.575 , 0.5125, 0.5125, 0.85  , 0.8   ,
       0.5125, 0.5125, 0.825 , 0.525 ]), 'mean_test_score': array([0.51  , 0.5075, 0.85  , 0.5075, 0.5075, 0.51  , 0.835 , 0.8475,
       0.51  , 0.7475, 0.8575, 0.6175, 0.5075, 0.5125, 0.8575, 0.85  ,
       0.51  , 0.5075, 0.85  , 0.5475]), 'std_test_score': array([0.005     , 0.01      , 0.03622844, 0.01      , 0.01      ,
       0.005     , 0.04062019, 0.04213075, 0.01658312, 0.03824265,
       0.02806243, 0.04444097, 0.01      , 0.0209165 , 0.01      ,
       0.03535534, 0.005     , 0.01      , 0.0209165 , 0.0242384 ]), 'rank_test_score': array([12, 16,  3, 16, 16, 12,  7,  6, 12,  8,  2,  9, 16, 11,  1,  3, 12,  
       16,  3, 10], dtype=int32)}
"""

"""
Q1. RandomizedSearchCV는 어떤 방식으로 최적의 하이퍼파라미터를 찾나요? GridSearchCV와 비교했을 때 어떤 장점이 있나요?
    
    방식: RandomizedSearchCV는 사용자가 정의한 하이퍼파라미터의 분포에서 
         n_iter 횟수만큼 무작위로 샘플링하여 조합을 생성합니다. 
         이렇게 생성된 각 조합에 대해 교차 검증을 수행하여 최적의 조합을 찾습니다.
    
    장점: 탐색 공간이 매우 넓거나 하이퍼파라미터가 많을 때, 
         GridSearchCV의 긴 학습 시간을 크게 단축할 수 있습니다. 
         GridSearchCV는 모든 조합을 테스트하지만, RandomizedSearchCV는 
         중요한 하이퍼파라미터 값에 더 많은 시간을 할애하여 효율적으로 탐색할 수 있다는 장점이 있습니다.

    GridSearch는 전수 조사
    RandomSearch는 표본 조사.

Q2. n_iter 매개변수는 RandomizedSearchCV에서 어떤 역할을 하나요? 이 값을 늘리거나 줄이면 어떤 영향을 미치나요?
    
    n_iter는 RandomizedSearchCV가 무작위로 샘플링하여 탐색할 하이퍼파라미터 조합의 개수를 지정하는 매개변수입니다.
    
    n_iter 값을 늘리면: 더 많은 조합을 탐색하게 되므로 최적의 하이퍼파라미터를 찾을 가능성이 높아지지만, 학습 시간도 함께 증가합니다.
    
    n_iter 값을 줄이면: 학습 시간이 단축되지만, 최적의 하이퍼파라미터를 찾지 못할 가능성이 커집니다.

Q3. 코드에서 uniform(loc, scale)과 같이 분포를 정의하는 이유는 무엇인가요? GridSearchCV처럼 값의 리스트를 사용할 수도 있나요?
    
    RandomizedSearchCV는 분포를 사용함으로써 연속적인 범위에서 무작위로 값을 샘플링할 수 있게 해줍니다. 
    이는 GridSearchCV가 미리 정해진 이산적인 값들만 탐색하는 것보다 더 넓은 탐색 공간을 효율적으로 커버할 수 있도록 돕습니다.
    
    네, 값의 리스트를 사용하는 것도 가능합니다. 예를 들어 {'C': [0.1, 1, 10]}과 같이 리스트를 전달하면, 
    RandomizedSearchCV는 이 리스트 내에서 무작위로 값을 선택하게 됩니다.

Q4. RandomizedSearchCV를 사용할 때 최적의 하이퍼파라미터를 찾지 못할 가능성이 있나요? 있다면, 이 단점을 보완하는 방법은 무엇인가요?
    
    네, RandomizedSearchCV는 무작위 탐색 방식이기 때문에 GridSearchCV처럼 모든 조합을 탐색하는 것은 아닙니다. 
    따라서 운이 나쁘면 최적의 조합을 찾지 못할 가능성이 있습니다.
    
    보완 방법:
        탐색 횟수(n_iter) 늘리기: n_iter를 늘려 탐색 횟수를 충분히 확보하면 최적의 조합을 찾을 확률이 높아집니다.
        2단계 튜닝: RandomizedSearchCV로 대략적인 최적의 하이퍼파라미터 범위를 찾은 후, 
        그 범위 주변을 GridSearchCV로 더 정교하게 탐색하는 방식을 사용하면 효율성과 정확도를 모두 높일 수 있습니다.

"""
