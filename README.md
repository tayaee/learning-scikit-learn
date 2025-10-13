# Learning AI/ML/DL/GenAI by Examples

**프로젝트 개요**

이 프로젝트는 머신러닝(ML), 딥러닝(DL), 그리고 생성형 AI(GenAI)의 핵심 개념과 주요 API를 학습하기 위한 로드맵을 제공합니다. `scikit-learn`을 시작으로, `TensorFlow`/`PyTorch` 기반의 딥러닝, `Transformer` 및 `BERT`를 활용한 자연어 처리, 그리고 강화학습과 AI 에이전트, 마지막으로 금융 및 보안 분야의 실용적인 응용 사례까지 폭넓게 다룹니다.

---

### **ML 학습 목차 (scikit-learn 기반)**

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
* `_2_5_1_regress_lasso_L1.py`: 규제 모델 `Lasso`
* `_2_5_1_regress_ridge_L2.py`: 규제 모델 `Ridge`

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

---

### **DL 학습 목차 (TensorFlow/PyTorch 기반)**

#### **Part 6: 딥러닝 기본 개념 및 프레임워크**
* `_6_1_dl_framework_install.py`: TensorFlow 또는 PyTorch 설치 및 기본 환경 설정
* `_6_2_tensors.py`: 텐서(Tensor)의 개념과 `numpy`와의 차이점
* `_6_3_autograd_concept.py`: 자동 미분(Autograd)의 원리 이해

#### **Part 7: 인공신경망 아키텍처 (Neural Network Architectures)**
* `_7_1_ann.py`: 간단한 신경망(ANN) 구현
* `_7_2_cnn.py`: 이미지 분류를 위한 합성곱 신경망(CNN) 구현
* `_7_3_rnn_lstm.py`: 시계열/자연어 처리를 위한 순환 신경망(RNN) 및 LSTM 구현

#### **Part 8: 딥러닝 모델 학습 및 평가**
* `_8_1_loss_optimizer.py`: 손실 함수(Loss function) 및 최적화기(Optimizer) 선택
* `_8_2_model_training_loop.py`: 학습 루프(Training Loop) 구현
* `_8_3_callbacks.py`: Early Stopping, Model Checkpointing 등 콜백 사용법
* `_8_4_transfer_learning.py`: 사전 학습된 모델(Pre-trained model) 활용 및 전이 학습(Transfer Learning)

#### **Part 9: 딥러닝 모델 배포 및 활용**
* `_9_1_model_save_load.py`: 모델 저장 및 로드
* `_9_2_inference_api.py`: 학습된 모델을 API로 배포하여 추론(Inference) 서비스 구축

---

### **GenAI 학습 목차 (생성형 AI)**

#### **Part 10: 생성형 AI 기본 모델**
* `_10_1_gan_vae_intro.py`: 생성적 적대 신경망(GAN) 및 변분 오토인코더(VAE) 개념 이해
* `_10_2_transformer_intro.py`: 트랜스포머(Transformer) 아키텍처의 핵심 원리

#### **Part 11: LLM (거대 언어 모델) 활용**
* `_11_1_bert_intro.py`: BERT를 활용한 자연어 처리 (전이 학습의 실제 사례)
* `_11_2_llm_api.py`: Gemini API, OpenAI API 등 주요 LLM API 사용
* `_11_3_prompt_engineering.py`: 효과적인 프롬프트 엔지니어링(Prompt Engineering) 기법 학습
* `_11_4_llm_fine_tuning.py`: 소규모 데이터셋으로 LLM 미세 조정(Fine-tuning)

#### **Part 12: GenAI 고급 기법 및 응용**
* `_12_1_rag.py`: RAG(검색 증강 생성)을 통한 답변 정확도 향상
* `_12_2_multimodal_api.py`: 이미지, 비디오 등 멀티모달 데이터 처리 API 활용
* `_12_3_gen_image_api.py`: Stable Diffusion, DALL-E 등 텍스트-이미지 생성 모델 API 사용

---

### **DL/GenAI 고급 학습 목차**

#### **Part 13: 강화학습 (Reinforcement Learning)**
* `_13_1_rl_intro.py`: 강화학습의 기본 개념 및 용어 (에이전트, 환경, 상태, 행동, 보상)
* `_13_2_q_learning.py`: Q-Learning 및 DQN(Deep Q-Network) 구현
* `_13_3_policy_gradients.py`: 정책 경사(Policy Gradient) 방법론 학습

#### **Part 14: AI 에이전트 및 멀티 에이전트 시스템**
* `_14_1_agentic_ai.py`: 에이전틱 AI(Agentic AI)의 개념 및 설계 원리
* `_14_2_a2a.py`: A2A(Agent-to-Agent) 통신 및 협력 모델 구축
* `_14_3_mcp.py`: MCP(Multi-agent Communication Protocol) 설계 및 적용

---

### **Part 15: 응용 프로젝트 및 실제 활용 사례**

* `_15_1_stock_prediction.py`: 시계열 학습 (RNN, LSTM) 기반 주가 예측 모델
* `_15_2_code_sast.py`: 정적 분석(SAST)을 위한 코드 취약점 예측 모델
* `_15_3_log_dast.py`: 동적 분석(DAST)을 위한 로그 기반 침입 탐지 및 이상 징후 감지
* `_15_4_nlp_for_security.py`: 자연어 처리(NLP)를 활용한 피싱 이메일 탐지
* `_15_5_cv_for_security.py`: 컴퓨터 비전(CV)을 활용한 CCTV 영상 내 이상 행동 감지

---

### **Part 16: 금융 및 IB 특화 AI 응용**

* `_16_1_algo_trading.py`: 알고리즘 트레이딩 및 HFT(고빈도 거래) 모델 구축 (강화학습, 시계열 예측 활용)
* `_16_2_risk_management.py`: AI 기반 신용 리스크 및 시장 리스크 관리 (신용 부도 예측, 이상 거래 탐지)
* `_16_3_nlp_for_research.py`: 금융 리포트 및 뉴스 자동 분석 (NLP, BERT 활용)
* `_16_4_compliance_ai.py`: AI 기반 컴플라이언스 및 규제 준수 자동화 (대화 감시, 거래 모니터링)
* `_16_5_back_office_automation.py`: AI 기반 백오피스 문서 자동화 (OCR, 계약서 분석, KYC/AML)