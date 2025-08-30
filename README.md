# 🏪 Rossmann Store Sales Prediction

## 📌 Problem Statement
Rossmann operates over 3,000 drug stores across Europe.  
The goal of this project is to **forecast daily sales** for stores up to six weeks in advance.  

Dataset: [Kaggle Rossmann Store Sales Competition](https://www.kaggle.com/competitions/rossmann-store-sales).  
Files:  
- `train.csv` (historical sales data).  
- `store.csv` (store metadata).  

Target variable: **Sales** (numeric, regression).

---

## ⚙️ Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Loaded `train.csv` and `store.csv`.  
- Checked holiday distributions (`StateHoliday`, `SchoolHoliday`).  
- Explored `StoreType` and `Assortment` categories.  
- Inspected missing values and distributions.  

### 2. Feature Engineering
- Merged sales data with store information.  
- Identified categorical vs numeric features.  
- Created input features (`X`) and target (`Sales`).  
- Train/Validation split applied.  

### 3. Preprocessing
- **Scaling**: `MinMaxScaler` for numeric features.  
- **Encoding**: `OneHotEncoder` for categorical features.  
- Built clean training and validation sets.

### 4. Baseline Models
- **Dumb Model**: Predicted mean sales for all rows (reference baseline).  
- **Base Model**: Simple linear baseline for comparison.

### 5. Linear Models
Trained multiple linear regression-based models:
- **Linear Regression** (OLS).  
- **Lasso Regression** (L1 regularization).  
- **Ridge Regression** (L2 regularization).  
- **ElasticNet** (L1 + L2).  
- **SGD Regressor**.  

### 6. Tree-Based Models
- **Decision Trees** → initial baseline tree model.  
- **Random Forest Regressor**:  
  - Used `RandomizedSearchCV` for hyperparameter tuning.  
  - Sampled 50k rows for efficient cross-validation.  
  - Tuned parameters like `n_estimators`, `max_depth`, `max_features`.  
  - Selected best model based on **RMSE**.  

### 7. Model Evaluation
Metric used: **RMSE (Root Mean Squared Error)** on Training and Validation sets.  

Findings:
- Linear models (OLS, Lasso, Ridge, ElasticNet, SGD) → moderate performance.  
- Random Forest → **best performance**, lowest validation RMSE.  
- Overfitting reduced after tuning via RandomizedSearchCV.  

---

## 📊 Key Insights
- Store-related features (`StoreType`, `Assortment`) and holidays significantly affect sales.  
- Linear models provide interpretability but weaker predictive power.  
- **Random Forest model achieved the least error on validation data**.  

---

## 🏆 Kaggle Leaderboard Result
- **Kaggle Score on test dataset**: `0.13`  
- **Public Leaderboard Rank**: `Top 2000` 


## 💻 Usage

### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
