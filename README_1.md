# Customer Churn Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-red?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)

---

## Project Overview

Customer churn — when a customer stops using a service — is one of the most critical business problems in the telecom industry. Acquiring a new customer costs **5x more** than retaining an existing one.

This project builds an end-to-end **machine learning pipeline** to predict which customers are likely to churn, enabling businesses to take proactive retention actions. The project covers data cleaning, exploratory data analysis (EDA), class imbalance handling using SMOTE, model training, evaluation, and model deployment using Pickle.

---

## Business Problem

**Goal:** Predict customer churn (Yes / No) for a telecom company based on customer demographics, account information, and service usage patterns.

**Impact:** Early identification of at-risk customers allows targeted retention campaigns, reducing revenue loss and improving customer lifetime value (CLV).

---

## Technical Skills Demonstrated

| Skill | Details |
|---|---|
| **Programming Language** | Python 3 |
| **Data Analysis & EDA** | Pandas, NumPy |
| **Data Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Class Imbalance Handling** | SMOTE (imblearn) |
| **Model Persistence** | Pickle |
| **Model Evaluation** | Accuracy, Confusion Matrix, Classification Report, Cross-Validation |
| **Feature Engineering** | Label Encoding, Feature Selection |
| **Development Environment** | Google Colab / Jupyter Notebook |

---

## Dataset Information

| Property | Details |
|---|---|
| **Dataset Name** | Telco Customer Churn |
| **Source** | [IBM Sample Dataset via Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Rows** | 7,043 customers |
| **Columns** | 21 features |
| **Target Variable** | `Churn` (Yes / No) |

### Key Features

- **Demographics** — gender, SeniorCitizen, Partner, Dependents
- **Account Info** — tenure, Contract type, PaymentMethod, MonthlyCharges, TotalCharges
- **Services** — PhoneService, MultipleLines, InternetService, OnlineSecurity, StreamingTV, StreamingMovies
- **Target** — Churn (binary classification)

---

## Project Workflow

```
Raw Data (CSV)
    │
    ▼
Data Loading & Understanding  →  df.shape: (7043, 21)
    │
    ▼
Exploratory Data Analysis (EDA)  →  Churn distribution, correlation heatmap,
    │                                feature distributions
    ▼
Data Preprocessing
    ├── Handle missing values
    ├── Label Encoding (categorical → numeric)
    └── Save encoders as encoders.pkl
    │
    ▼
Class Imbalance Handling  →  SMOTE (Synthetic Minority Oversampling Technique)
    │
    ▼
Train / Test Split  →  80% train, 20% test
    │
    ▼
Model Training & Cross-Validation (5-Fold)
    ├── Decision Tree Classifier
    ├── Random Forest Classifier  ← Best model
    └── XGBoost Classifier
    │
    ▼
Model Evaluation  →  Accuracy, Confusion Matrix, Classification Report
    │
    ▼
Model Deployment  →  Save as customer_churn_model.pkl
    │
    ▼
Prediction on New Customer Input
```

---

## Machine Learning Models

| Model | Cross-Validation Accuracy |
|---|---|
| Decision Tree | 78% |
| **Random Forest** | **84% (Best)** |
| XGBoost | Evaluated |

### Best Model — Random Forest Classifier

```
Test Set Accuracy : 77.86%

Confusion Matrix:
              Predicted No   Predicted Yes
Actual No         878            158
Actual Yes        154            219

Classification Report:
              Precision   Recall   F1-Score
No Churn         0.85       0.85     0.85
Churn            0.58       0.59     0.58
```

> **Note:** Class imbalance in churn datasets is a known challenge. SMOTE was applied on training data to improve minority class (churn) recall.

---

## Sample Prediction

```python
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'tenure': 1,
    'Contract': 'Month-to-month',
    'MonthlyCharges': 29.85,
    ...
}

# Output
Prediction: No Churn
Prediction Probability: [0.78  0.22]
```

---

## Folder Structure

```
customer-churn-prediction/
│
├── Customer_Churn_Prediction_using_ML.ipynb   ← Main notebook (run this)
├── README.md                                   ← Project documentation
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  ← Dataset (download from Kaggle)
│
├── models/
│   ├── customer_churn_model.pkl               ← Trained Random Forest model
│   └── encoders.pkl                           ← Saved label encoders
│
└── images/
    ├── churn_distribution.png                 ← EDA charts
    ├── correlation_heatmap.png
    └── confusion_matrix.png
```

---

## How to Run

### Option 1 — Google Colab (recommended, no setup needed)

1. Open [Google Colab](https://colab.research.google.com)
2. Click `File` → `Upload notebook` → upload `Customer_Churn_Prediction_using_ML.ipynb`
3. Upload the dataset CSV when prompted
4. Click `Runtime` → `Run all`

### Option 2 — Local (Jupyter Notebook)

**Step 1 — Clone the repository**
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the `data/` folder.

**Step 4 — Run the notebook**
```bash
jupyter notebook Customer_Churn_Prediction_using_ML.ipynb
```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
pickle5
jupyter
```

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn notebook
```

---

## Key Insights from EDA

- Customers on **Month-to-month contracts** have the highest churn rate (~42%)
- Customers with **tenure < 12 months** are most likely to churn
- **Fiber optic** internet service users churn more than DSL users
- Customers **without online security** show significantly higher churn
- Higher **monthly charges** correlate with increased churn probability

---

## Future Improvements

- [ ] Hyperparameter tuning with GridSearchCV / RandomizedSearchCV
- [ ] Add more advanced models (LightGBM, CatBoost)
- [ ] Build a Streamlit web app for live churn prediction
- [ ] Add SHAP values for model explainability
- [ ] Deploy model as REST API using Flask

---

## Connect with Me

**LinkedIn:** [linkedin.com/in/yourname](https://linkedin.com/in/yourname)  
**GitHub:** [github.com/yourusername](https://github.com/yourusername)  
**Email:** yourname@gmail.com

---

## License

This project is licensed under the MIT License — feel free to use and modify it.

---

*Built with Python | Scikit-learn | XGBoost | Pandas | Matplotlib | SMOTE*
