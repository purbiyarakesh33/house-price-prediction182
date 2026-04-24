#  House Price Predictor

Predicts house prices using an Ensemble of XGBoost, Random Forest & Ridge Regression | Ames Housing Dataset | Streamlit App

##  Live Demo
**[https://house-price-predictor182.streamlit.app](https://house-price-prediction182.streamlit.app)**  

---

##  Project Overview

| Item | Detail |
|------|--------|
| Dataset | Ames Housing Dataset (2,930 houses, 80 features) |
| Problem | Predict house sale price (Regression) |
| Best Model | Ensemble (XGBoost + Random Forest + Ridge) |
| Final R² | 0.8441 |
| RMSE (log scale) | 0.1448 |

---

##  Steps Followed

1. **Data Cleaning** — null handling, one-hot encoding, dropping high-null columns
2. **EDA & Correlation Analysis** — Pearson correlation, feature selection (|corr| ≥ 0.5)
3. **Outlier Removal** — IQR method on SalePrice
4. **Feature Engineering** — 12 new features created:
   - `TotalSF` = Basement + 1st Floor + Living Area
   - `Qual_x_TotalSF` = Overall Quality × Total SF (top predictor)
   - `HouseAge`, `RemodAge`, `Was_Remodeled`
   - `HasGarage`, `GarageEfficiency`, `HasMasVnr`
   - `KitchenScore`, `Qual_x_Area`
   - Log transform on SalePrice target
5. **Train / Val / Test Split** — 60% / 20% / 20% (no data leakage)
6. **Model Training & Hyperparameter Tuning** — GridSearchCV (cv=5) on all models
7. **Bias/Variance Analysis** — Learning curves for XGBoost & Random Forest
8. **Optimized Ensemble** — scipy minimize to find mathematically optimal weights
9. **Deployment** — Streamlit web app

---

##  Results
| Model | Test R² | Test RMSE |
|-------|---------|-----------|
| Ridge (Baseline) | 0.8266 | 0.1527 |
| Random Forest | 0.8295 | 0.1514 |
| XGBoost | 0.8410 | 0.1462 |
| Neural Network | 0.4625 | 0.2688 |
| **Ensemble (Optimized)** | **0.8441** | **0.1448** |

> Neural Network underperforms on this dataset size (~1,800 rows after cleaning) — gradient boosting methods are better suited for small tabular data.

> Random Forest shows high variance even after GridSearchCV tuning — the persistent train/val gap indicates a data size limitation, not a tuning problem. This is why XGBoost with built-in regularization outperforms it.

---

##  Feature Importance (XGBoost)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Qual_x_TotalSF | 33.7% |
| 2 | Qual_x_Area | 16.1% |
| 3 | Overall Qual | 7.4% |
| 4 | TotalSF | 4.7% |
| 5 | Garage Cars | 4.2% |

---

## 🛠 Tech Stack

- **Language:** Python
- **ML:** Scikit-learn, XGBoost
- **Data:** Pandas, Numpy, Scipy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit

---

##  How to Run Locally

```bash
git clone https://github.com/purbiyarakesh33/house-price-predictor.git
cd house-price-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

##  Project Structure

```
house-price-predictor/
├── app.py                  # Streamlit web app
├── 1.py                    # Full ML pipeline
├── model_xgb.pkl           # XGBoost model
├── model_rf.pkl            # Random Forest model
├── model_ridge.pkl         # Ridge model
├── scaler_final.pkl        # StandardScaler
├── feature_cols.pkl        # Feature column list
├── ensemble_weights.pkl    # Optimized ensemble weights
├── requirements.txt
└── README.md
```
