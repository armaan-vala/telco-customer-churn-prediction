# Telco Customer Churn Prediction 📉🤖

An end-to-end Machine Learning pipeline built to predict customer churn in the telecommunications industry. This project focuses on rigorous data preprocessing, handling class imbalance, and comparing linear vs. ensemble supervised models using industry-standard engineering practices.

---

## 🚀 Project Overview

Instead of jumping straight into complex algorithms, this project follows a disciplined machine learning lifecycle:

1. **Data Auditing & Cleaning** (Handling hidden blank spaces and data types)
2. **Preprocessing & Feature Engineering** (One-hot encoding and scaling)
3. **Baseline Benchmarking** (Proving why naive accuracy fails under class imbalance)
4. **Supervised Modeling & Evaluation** (Logistic Regression vs. Random Forest, analyzed via Confusion Matrix, Precision, Recall, F1-Score, and ROC-AUC curves)

---

## 🛠️ What We Did (Step-by-Step Workflow)

* **Data Cleaning & Preprocessing:** Audited the dataset, coerced hidden blank spaces in `TotalCharges` to `NaN`, dropped missing rows cleanly, and encoded the target variable (`Churn`) into binary format.
* **One-Hot Encoding (`drop_first=True`):** Converted categorical text columns into binary indicator features while successfully avoiding the dummy variable trap (multicollinearity).
* **Stratified Train-Test Split:** Divided the data into an 80% training set and 20% testing set while preserving the exact 73%-26% class distribution ratio.
* **Feature Scaling (`StandardScaler`):** Standardized numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`) to prevent feature magnitude dominance in Logistic Regression, ensuring zero data leakage by fitting strictly on the training set.
* **Baseline Benchmark Model:** Established a performance floor using a majority-class `DummyClassifier`, highlighting why high accuracy can be dangerously misleading when class imbalance is present.
* **Supervised Learning Models:**
  * **Logistic Regression:** Trained a linear decision boundary using the **Sigmoid function** to output probabilities and classified using a standard **0.5 threshold**.
  * **Random Forest Classifier:** Trained an ensemble of **Decision Trees** using **Majority Voting** to capture non-linear feature interactions without requiring scaling.
* **Comprehensive Evaluation Metrics:** Evaluated models beyond basic accuracy by analyzing **Confusion Matrices**, **Precision**, **Recall**, **F1-Score**, and plotting **ROC-AUC Curves**.

---

## 📊 Key Results & Findings

* **Logistic Regression (AUC = 0.836)** outperformed **Random Forest (AUC = 0.816)** on this dataset, proving that linear relationships dominated the feature space and complex ensembles were prone to minor overfitting.
* Addressed the **Class Imbalance Problem** by prioritizing **Recall and F1-Score** over naive accuracy, ensuring the model successfully catches actual churners rather than just guessing the majority class.

---

## 📂 Project Structure

```text
├── churn-dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── churn_analysis.ipynb          # containing step-by-step code    
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation

```

## ⚙️ How to Run Locally

Follow these steps to set up and run the project on your machine.

### 1. Clone the Repository

```bash
git clone https://github.com/armaan-vala/telco-customer-churn-prediction.git
```

### 2. generate isolated enviroment 

### 3. Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
