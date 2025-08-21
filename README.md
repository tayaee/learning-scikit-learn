### scikit-learn API 사용법 학습 목차

#### **Part 1: Scikit-learn 기본 개념 및 API**
* `_1_1_preprocess_standardscaler.py`: 데이터 스케일링 `StandardScaler`
* `_1_2_preprocess_simpleimputer.py`: 결측값 처리 `SimpleImputer`
* `_1_3_preprocess_onehotencoder.py`: 범주형 데이터 인코딩 `OneHotEncoder`
* `_1_4_preprocess_minmaxscaler.py`: 데이터 정규화 `MinMaxScaler`

#### **Part 2: 지도 학습 (Supervised Learning)**
* `_2_1_classify_randomforest.py`: 분류 모델 `RandomForestClassifier`
* `_2_2_classify_svc.py`: 분류 모델 `sklearn.svm.SVC`
* `_2_3_regress_linear.py`: 회귀 모델 `LinearRegression`
* `_2_4_regress_randomforest.py`: 회귀 모델 `RandomForestRegressor`
* `_2_5_regress_lasso_ridge.py`: 규제 모델 `Lasso`, `Ridge`

#### **Part 3: 비지도 학습 (Unsupervised Learning)**
* `_3_1_cluster_kmeans.py`: 군집 모델 `KMeans`
* `_3_2_cluster_dbscan.py`: 군집 모델 `DBSCAN`
* `_3_3_dimred_pca.py`: 차원 축소 `PCA`
* `_3_4_dimred_tsne.py`: 차원 축소 `t-SNE`

#### **Part 4: 모델 평가 및 최적화**
* `_4_1_model_selection_cv.py`: 모델 검증 `cross_val_score`, `KFold`
* `_4_2_metrics.py`: 모델 성능 지표 `accuracy`, `precision`, `recall`, `f1_score`
* `_4_3_hyperparameter_tuning_gridsearch.py`: 하이퍼파라미터 튜닝 `GridSearchCV`
* `_4_4_hyperparameter_tuning_randomsearch.py`: 하이퍼파라미터 튜닝 `RandomizedSearchCV`

#### **Part 5: 고급 기능 및 워크플로우**
* `_5_1_pipeline.py`: 파이프라인 `Pipeline`
* `_5_2_model_serialization.py`: 모델 저장/로드 `joblib`
