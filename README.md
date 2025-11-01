# Learning AI/ML/DL/GenAI by Examples

**Project Overview**

This repository documents the usage of key APIs in Machine Learning (ML), Deep Learning (DL), and Generative AI (GenAI) while learning their core concepts.

---

### **ML Learning Roadmap (Based on scikit-learn)**

#### **Part 1: Scikit-learn Core Concepts and APIs**
| Filename | Description |
| :--- | :--- |
| `_1_1_preprocess_standardscaler.py` | Data Scaling **`StandardScaler`** |
| `_1_2_preprocess_simpleimputer.py` | Missing Value Handling **`SimpleImputer`** |
| `_1_3_preprocess_onehotencoder.py` | Categorical Data Encoding **`OneHotEncoder`** |
| `_1_4_preprocess_minmaxscaler.py` | Data Normalization **`MinMaxScaler`** |

#### **Part 2: Supervised Learning**
| Filename | Description |
| :--- | :--- |
| `_2_1_classify_randomforest.py` | Classification Model **`RandomForestClassifier`** |
| `_2_2_classify_svc.py` | Classification Model **`sklearn.svm.SVC`** |
| `_2_3_regress_linear.py` | Regression Model **`LinearRegression`** |
| `_2_4_regress_randomforest.py` | Regression Model **`RandomForestRegressor`** |
| `_2_5_1_regress_lasso_L1.py` | Regularization Model **`Lasso`** |
| `_2_5_1_regress_ridge_L2.py` | Regularization Model **`Ridge`** |

#### **Part 3: Unsupervised Learning**
| Filename | Description |
| :--- | :--- |
| `_3_1_cluster_kmeans.py` | Clustering Model **`KMeans`** |
| `_3_2_cluster_dbscan.py` | Clustering Model **`DBSCAN`** |
| `_3_3_dimred_pca.py` | Dimensionality Reduction **`PCA`** |
| `_3_4_dimred_tsne.py` | Dimensionality Reduction **`t-SNE`** |

#### **Part 4: Model Evaluation and Optimization**
| Filename | Description |
| :--- | :--- |
| `_4_1_model_selection_cv.py` | Model Validation **`cross_val_score`**, **`KFold`** |
| `_4_2_metrics.py` | Model Performance Metrics **`accuracy`**, **`precision`**, **`recall`**, **`f1_score`** |
| `_4_3_hyperparameter_tuning_gridsearch.py` | Hyperparameter Tuning **`GridSearchCV`** |
| `_4_4_hyperparameter_tuning_randomsearch.py` | Hyperparameter Tuning **`RandomizedSearchCV`** |

#### **Part 5: Advanced Features and Workflow**
| Filename | Description |
| :--- | :--- |
| `_5_1_pipeline.py` | Pipeline **`Pipeline`** |
| `_5_2_model_serialization.py` | Model Save/Load **`joblib`** |

---

### **DL Learning Roadmap (Based on TensorFlow/PyTorch)**

#### **Part 6: Deep Learning Fundamentals and Frameworks**
| Filename | Description |
| :--- | :--- |
| `_6_1_dl_framework_install.py` | TensorFlow or PyTorch Installation and Basic Setup |
| `_6_2_tensors.py` | Concept of **Tensors** and Differences from **`numpy`** |
| `_6_3_autograd_concept.py` | Understanding the Principle of **Automatic Differentiation (Autograd)** |

#### **Part 7: Neural Network Architectures**
| Filename | Description |
| :--- | :--- |
| `_7_1_ann.py` | Implementation of a Simple **Artificial Neural Network (ANN)** |
| `_7_2_cnn.py` | Implementation of a **Convolutional Neural Network (CNN)** for Image Classification |
| `_7_3_rnn_lstm.py` | Implementation of **Recurrent Neural Networks (RNN)** and **LSTM** for Time Series/NLP |

#### **Part 8: Deep Learning Model Training and Evaluation**
| Filename | Description |
| :--- | :--- |
| `_8_1_loss_optimizer.py` | Selection of **Loss function** and **Optimizer** |
| `_8_2_model_training_loop.py` | Implementation of the **Training Loop** |
| `_8_3_callbacks.py` | Usage of Callbacks like **Early Stopping**, **Model Checkpointing**, etc. |
| `_8_4_transfer_learning.py` | Utilizing **Pre-trained models** and **Transfer Learning** |

#### **Part 9: Deep Learning Model Deployment and Application**
| Filename | Description |
| :--- | :--- |
| `_9_1_model_save_load.py` | Model Saving and Loading |
| `_9_2_inference_api.py` | Building an **Inference Service** by deploying a trained model as an API |

---

### **GenAI Learning Roadmap (Generative AI)**

#### **Part 10: Generative AI Fundamental Models**
| Filename | Description |
| :--- | :--- |
| `_10_1_gan_vae_intro.py` | Understanding the Concepts of **Generative Adversarial Networks (GAN)** and **Variational Autoencoders (VAE)** |
| `_10_2_transformer_intro.py` | Core Principles of the **Transformer Architecture** |

#### **Part 11: LLM (Large Language Model) Application**
| Filename | Description |
| :--- | :--- |
| `_11_1_bert_intro.py` | Natural Language Processing using **BERT** (Practical example of Transfer Learning) |
| `_11_2_llm_api.py` | Using Major LLM APIs such as **Gemini API**, **OpenAI API** |
| `_11_3_prompt_engineering.py` | Learning Effective **Prompt Engineering** Techniques |
| `_11_4_llm_fine_tuning.py` | **Fine-tuning** LLMs with Small Datasets |

#### **Part 12: GenAI Advanced Techniques and Applications**
| Filename | Description |
| :--- | :--- |
| `_12_1_rag.py` | Improving Answer Accuracy with **RAG (Retrieval-Augmented Generation)** |
| `_12_2_multimodal_api.py` | Utilizing APIs for **Multimodal Data Processing** (Image, Video, etc.) |
| `_12_3_gen_image_api.py` | Using Text-to-Image Generation Model APIs such as **Stable Diffusion**, **DALL-E** |

---

### **DL/GenAI Advanced Learning Roadmap**

#### **Part 13: Reinforcement Learning (RL)**
| Filename | Description |
| :--- | :--- |
| `_13_1_rl_intro.py` | Fundamental Concepts and Terminology of RL (**Agent, Environment, State, Action, Reward**) |
| `_13_2_q_learning.py` | Implementation of **Q-Learning** and **DQN (Deep Q-Network)** |
| `_13_3_policy_gradients.py` | Learning the **Policy Gradient** Methodology |

#### **Part 14: AI Agent and Multi-Agent Systems**
| Filename | Description |
| :--- | :--- |
| `_14_1_agentic_ai.py` | Concepts and Design Principles of **Agentic AI** |
| `_14_2_a2a.py` | Building **A2A (Agent-to-Agent)** Communication and Collaborative Models |
| `_14_3_mcp.py` | Designing and Applying **MCP (Multi-agent Communication Protocol)** |

---

### **Part 15: Application Projects and Real-World Use Cases**

| Filename | Description |
| :--- | :--- |
| `_15_1_stock_prediction.py` | Stock Price Prediction Model based on Time Series Learning (**RNN, LSTM**) |
| `_15_2_code_sast.py` | Code Vulnerability Prediction Model for **Static Analysis (SAST)** |
| `_15_3_log_dast.py` | Log-based Intrusion Detection and Anomaly Detection for **Dynamic Analysis (DAST)** |
| `_15_4_nlp_for_security.py` | Phishing Email Detection using **Natural Language Processing (NLP)** |
| `_15_5_cv_for_security.py` | Abnormal Behavior Detection in CCTV Footage using **Computer Vision (CV)** |

---

### **Part 16: AI Applications Specialized for Finance and IB**

| Filename | Description |
| :--- | :--- |
| `_16_1_algo_trading.py` | Building Algorithmic Trading and **HFT (High-Frequency Trading)** Models |
| `_16_2_risk_management.py` | AI-based **Credit Risk and Market Risk Management** |
| `_16_3_nlp_for_research.py` | Automatic Analysis of Financial Reports and News (using **NLP, BERT**) |
| `_16_4_compliance_ai.py` | AI-based **Compliance and Regulatory Automation** |
| `_16_5_back_office_automation.py` | AI-based **Back-Office Document Automation** (**OCR, Contract Analysis, KYC/AML**) |