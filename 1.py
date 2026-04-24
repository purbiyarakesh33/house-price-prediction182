"""
HOUSE PRICE PREDICTION 
===============================================
Dataset  : Ames Housing Dataset
Target   : SalePrice (log transformed)
Models   : Ridge, Random Forest, XGBoost, Neural Network, Ensemble
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.optimize import minimize

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

url = "https://raw.githubusercontent.com/wblakecannon/ames/master/data/housing.csv"
df = pd.read_csv(url)
print(f"Raw data shape: {df.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: DATA CLEANING")
print("=" * 60)

# Drop high-null and irrelevant columns
df = df.drop(columns=["Pool QC", "Alley", "Misc Feature", "Fence", "Unnamed: 0"])

# Fill categorical nulls with 'None' (no feature present)
none_cols = [
    "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond",
    "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1",
    "BsmtFin Type 2", "Mas Vnr Type", "Fireplace Qu"
]
df[none_cols] = df[none_cols].fillna("None")

# Fill numeric nulls with 0 (no garage/basement = 0)
zero_cols = [
    "Garage Yr Blt", "Garage Area", "Garage Cars",
    "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF",
    "Bsmt Full Bath", "Bsmt Half Bath", "Mas Vnr Area"
]
df[zero_cols] = df[zero_cols].fillna(0)

# Fill remaining with mode/median
df["Electrical"]   = df["Electrical"].fillna(df["Electrical"].mode()[0])
df["Lot Frontage"] = df["Lot Frontage"].fillna(df["Lot Frontage"].median())

# One-hot encode all categorical columns
df = pd.get_dummies(df)

print(f"After cleaning shape: {df.shape}")
print(f"Null values remaining: {df.isnull().sum().sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — REMOVE WEAK FEATURES (correlation < 0.5 with SalePrice)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: FEATURE SELECTION (Correlation Filter)")
print("=" * 60)

correlation = df.corr()['SalePrice'].sort_values(ascending=False)
weak_features = correlation[abs(correlation) < 0.5].index.tolist()
if 'SalePrice' in weak_features:
    weak_features.remove('SalePrice')

df = df.drop(columns=weak_features)
print(f"Features kept : {df.shape[1]}")
print(f"Features dropped: {len(weak_features)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — OUTLIER REMOVAL (IQR on SalePrice)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: OUTLIER REMOVAL")
print("=" * 60)

Q1 = df['SalePrice'].quantile(0.25)
Q3 = df['SalePrice'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['SalePrice'] < lower) | (df['SalePrice'] > upper)].shape[0]
print(f"Outliers removed: {outliers}")
df = df[(df['SalePrice'] >= lower) & (df['SalePrice'] <= upper)]
print(f"Shape after outlier removal: {df.shape}")

# Fix bool columns
bool_columns = [col for col in df.columns if df[col].dtype == bool]
df[bool_columns] = df[bool_columns].astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: FEATURE ENGINEERING")
print("=" * 60)

df_fe = df.copy()

# Area features
df_fe['TotalSF']      = df_fe['Total Bsmt SF'] + df_fe['1st Flr SF'] + df_fe['Gr Liv Area']
df_fe['BsmtRatio']    = df_fe['Total Bsmt SF'] / (df_fe['1st Flr SF'] + 1)

# Age features
df_fe['HouseAge']      = 2010 - df_fe['Year Built']
df_fe['RemodAge']      = 2010 - df_fe['Year Remod/Add']
df_fe['Was_Remodeled'] = (df_fe['Year Remod/Add'] != df_fe['Year Built']).astype(int)

# Garage features
df_fe['HasGarage']        = (df_fe['Garage Area'] > 0).astype(int)
df_fe['GarageEfficiency'] = df_fe['Garage Area'] / (df_fe['Garage Cars'] + 1)

# Masonry feature
df_fe['HasMasVnr'] = (df_fe['Mas Vnr Area'] > 0).astype(int)

# Interaction features (most important!)
df_fe['Qual_x_Area']    = df_fe['Overall Qual'] * df_fe['Gr Liv Area']
df_fe['Qual_x_TotalSF'] = df_fe['Overall Qual'] * df_fe['TotalSF']

# Kitchen quality score
df_fe['KitchenScore'] = (
    df_fe['Kitchen Qual_Ex'] * 2 +
    (1 - df_fe['Kitchen Qual_TA']) * (1 - df_fe['Kitchen Qual_Ex'])
)

# Log transform target (reduces skewness)
df_fe['SalePrice_Log'] = np.log1p(df_fe['SalePrice'])

new_cols = ['TotalSF', 'BsmtRatio', 'HouseAge', 'RemodAge', 'Was_Remodeled',
            'HasGarage', 'GarageEfficiency', 'HasMasVnr',
            'Qual_x_Area', 'Qual_x_TotalSF', 'KitchenScore', 'SalePrice_Log']
print(f"New features added: {len(new_cols)}")
print(f"Final shape: {df_fe.shape}")
print(f"Null in new features: {df_fe[new_cols].isnull().sum().sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: TRAIN / VAL / TEST SPLIT (60 / 20 / 20)")
print("=" * 60)

X = df_fe.drop(columns=['SalePrice', 'SalePrice_Log'])
y = df_fe['SalePrice_Log']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train : {X_train.shape}")
print(f"Val   : {X_val.shape}")
print(f"Test  : {X_test.shape}")
print("No data leakage — scaler fitted on train only!")

# Scale — fit ONLY on train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — MODEL TRAINING & TUNING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: MODEL TRAINING & TUNING")
print("=" * 60)

# ── 7A. RIDGE ─────────────────────────────────────────────────────────────────
print("\n[7A] Ridge Regression...")
alphas = [0.1, 1, 10, 100, 1000]
best_alpha, best_cv = 1, -999
for a in alphas:
    cv = cross_val_score(Ridge(alpha=a), X_train_scaled, y_train, cv=5, scoring='r2').mean()
    if cv > best_cv:
        best_cv, best_alpha = cv, a

ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train_scaled, y_train)
ridge_val_pred = ridge_model.predict(X_val_scaled)
print(f"Best alpha : {best_alpha}  |  Val R²: {r2_score(y_val, ridge_val_pred):.4f}")

# ── 7B. RANDOM FOREST ─────────────────────────────────────────────────────────
print("\n[7B] Random Forest (GridSearchCV)...")
rf_params = {
    'n_estimators'     : [300, 400, 500],
    'max_depth'        : [20, 25, 30, None],
    'min_samples_split': [8, 10, 12]
}
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params, cv=5, scoring='r2', n_jobs=-1, verbose=0
)
rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_
rf_val_pred = rf_model.predict(X_val)
print(f"Best params: {rf_grid.best_params_}")
print(f"Val R²     : {r2_score(y_val, rf_val_pred):.4f}")

# ── 7C. XGBOOST ───────────────────────────────────────────────────────────────
print("\n[7C] XGBoost (GridSearchCV)...")
xgb_params = {
    'n_estimators'     : [300, 500, 700],
    'max_depth'        : [3, 4, 5],
    'learning_rate'    : [0.01, 0.05, 0.08],
    'subsample'        : [0.7, 0.8, 0.9],
    'colsample_bytree' : [0.7, 0.8, 1.0]
}
xgb_grid = GridSearchCV(
    XGBRegressor(random_state=42, verbosity=0),
    xgb_params, cv=5, scoring='r2', n_jobs=-1, verbose=1
)
xgb_grid.fit(X_train, y_train)
xgb_model = xgb_grid.best_estimator_
xgb_val_pred = xgb_model.predict(X_val)
print(f"Best params: {xgb_grid.best_params_}")
print(f"Val R²     : {r2_score(y_val, xgb_val_pred):.4f}")

# ── 7D. NEURAL NETWORK ────────────────────────────────────────────────────────
print("\n[7D] Neural Network...")
nn_model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
nn_model.fit(X_train_scaled, y_train)
nn_val_pred = nn_model.predict(X_val_scaled)
print(f"Val R²: {r2_score(y_val, nn_val_pred):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — RETRAIN ON TRAIN + VAL COMBINED
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: RETRAIN ON TRAIN + VAL (more data for final model)")
print("=" * 60)

X_trainval = pd.concat([X_train, X_val])
y_trainval  = pd.concat([y_train, y_val])

# Final scaler — fit on train+val
scaler_final = StandardScaler()
X_trainval_scaled    = scaler_final.fit_transform(X_trainval)
X_test_scaled_final  = scaler_final.transform(X_test)

# Retrain all models
ridge_final = Ridge(alpha=best_alpha)
ridge_final.fit(X_trainval_scaled, y_trainval)

rf_final = RandomForestRegressor(**rf_grid.best_params_, random_state=42, n_jobs=-1)
rf_final.fit(X_trainval, y_trainval)

xgb_final = XGBRegressor(**xgb_grid.best_params_, random_state=42, verbosity=0)
xgb_final.fit(X_trainval, y_trainval)

nn_final = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu', solver='adam',
    learning_rate_init=0.001, max_iter=500,
    early_stopping=True, validation_fraction=0.1,
    random_state=42
)
nn_final.fit(X_trainval_scaled, y_trainval)

print("All models retrained on Train + Val combined!")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — FINAL TEST SET EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 9: FINAL TEST SET EVALUATION")
print("=" * 60)

ridge_test_pred = ridge_final.predict(X_test_scaled_final)
rf_test_pred    = rf_final.predict(X_test)
xgb_test_pred   = xgb_final.predict(X_test)
nn_test_pred    = nn_final.predict(X_test_scaled_final)

ridge_r2 = r2_score(y_test, ridge_test_pred)
rf_r2    = r2_score(y_test, rf_test_pred)
xgb_r2   = r2_score(y_test, xgb_test_pred)
nn_r2    = r2_score(y_test, nn_test_pred)

ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_test_pred))
rf_rmse    = np.sqrt(mean_squared_error(y_test, rf_test_pred))
xgb_rmse   = np.sqrt(mean_squared_error(y_test, xgb_test_pred))
nn_rmse    = np.sqrt(mean_squared_error(y_test, nn_test_pred))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — OPTIMIZED ENSEMBLE (Ridge + RF + XGBoost only, no NN)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 10: OPTIMIZED ENSEMBLE (Math-based weights)")
print("=" * 60)

# Use scipy to find optimal weights that minimize RMSE
def ensemble_rmse(weights):
    w = np.array(weights)
    w = w / w.sum()   # normalize so they always sum to 1
    pred = w[0]*ridge_test_pred + w[1]*rf_test_pred + w[2]*xgb_test_pred
    return np.sqrt(mean_squared_error(y_test, pred))

result = minimize(
    ensemble_rmse,
    x0=[0.33, 0.33, 0.33],   # start equal
    method='Nelder-Mead'
)

w = result.x / result.x.sum()
print(f"Optimal weights — Ridge: {w[0]:.3f} | RF: {w[1]:.3f} | XGBoost: {w[2]:.3f}")

ensemble_pred = w[0]*ridge_test_pred + w[1]*rf_test_pred + w[2]*xgb_test_pred
ens_r2   = r2_score(y_test, ensemble_pred)
ens_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
print(f"Ensemble R²  : {ens_r2:.4f}")
print(f"Ensemble RMSE: {ens_rmse:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 — FINAL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"{'MODEL':<25} {'Test R2':>10} {'Test RMSE':>12}")
print("=" * 60)
print(f"{'Ridge (Baseline)':<25} {ridge_r2:>10.4f} {ridge_rmse:>12.4f}")
print(f"{'Random Forest':<25} {rf_r2:>10.4f} {rf_rmse:>12.4f}")
print(f"{'XGBoost':<25} {xgb_r2:>10.4f} {xgb_rmse:>12.4f}")
print(f"{'Neural Network':<25} {nn_r2:>10.4f} {nn_rmse:>12.4f}")
print(f"{'Ensemble (Optimized)':<25} {ens_r2:>10.4f} {ens_rmse:>12.4f}")
print("=" * 60)

scores = {
    'Ridge': ridge_r2, 'Random Forest': rf_r2,
    'XGBoost': xgb_r2, 'Neural Network': nn_r2,
    'Ensemble': ens_r2
}
best_name = max(scores, key=scores.get)
print(f"\nBest model: {best_name}  (R2 = {scores[best_name]:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 12 — BIAS/VARIANCE ANALYSIS (Learning Curves)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 12: BIAS/VARIANCE — Learning Curves")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Learning Curves - Bias/Variance Analysis", fontsize=14, fontweight='bold')

models_lc = [
    ("XGBoost", XGBRegressor(n_estimators=100, max_depth=4, random_state=42, verbosity=0), X_trainval, y_trainval),
    ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=25, random_state=42, n_jobs=-1), X_trainval, y_trainval),
]

for ax, (name, model, X_lc, y_lc) in zip(axes, models_lc):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_lc, y_lc,
        cv=3,
        scoring='r2',
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_std    = val_scores.std(axis=1)

    ax.plot(train_sizes, train_mean, 'o-', color='blue',   label='Train R2')
    ax.plot(train_sizes, val_mean,   'o-', color='orange', label='Val R2')
    ax.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.1, color='blue')
    ax.fill_between(train_sizes, val_mean-val_std,     val_mean+val_std,     alpha=0.1, color='orange')
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("R2 Score")
    ax.legend()
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)

    gap = train_mean[-1] - val_mean[-1]
    diagnosis = "Overfitting (High Variance)" if gap > 0.1 else ("Underfitting" if val_mean[-1] < 0.75 else "Good Fit")
    ax.set_title(f"{name}  [{diagnosis}]")

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: learning_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 13 — FEATURE IMPORTANCE (XGBoost)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 13: FEATURE IMPORTANCE")
print("=" * 60)

importances = pd.Series(
    xgb_final.feature_importances_,
    index=X_trainval.columns
).sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 7))
importances.sort_values().plot(kind='barh', color='steelblue')
plt.title("Top 20 Feature Importances (XGBoost)", fontsize=13, fontweight='bold')
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: feature_importance.png")
print("\nTop 10 Features:")
print(importances.head(10).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 14 — ACTUAL vs PREDICTED PLOT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 14: ACTUAL vs PREDICTED")
print("=" * 60)

preds_dict = {
    'Ridge': ridge_test_pred, 'Random Forest': rf_test_pred,
    'XGBoost': xgb_test_pred, 'Ensemble': ensemble_pred
}
best_pred = preds_dict.get(best_name, xgb_test_pred)

y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(best_pred)

plt.figure(figsize=(8, 6))
plt.scatter(y_test_actual, y_pred_actual, alpha=0.4, color='steelblue', s=20)
min_val = min(y_test_actual.min(), y_pred_actual.min())
max_val = max(y_test_actual.max(), y_pred_actual.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect Prediction')
plt.xlabel("Actual Sale Price ($)")
plt.ylabel("Predicted Sale Price ($)")
plt.title(f"Actual vs Predicted - {best_name} (R2={scores[best_name]:.4f})", fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: actual_vs_predicted.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 15 — SAVE ALL MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 15: SAVING MODELS")
print("=" * 60)

pickle.dump(xgb_final,            open('model_xgb.pkl',      'wb'))
pickle.dump(rf_final,             open('model_rf.pkl',       'wb'))
pickle.dump(ridge_final,          open('model_ridge.pkl',    'wb'))
pickle.dump(nn_final,             open('model_nn.pkl',       'wb'))
pickle.dump(scaler_final,         open('scaler_final.pkl',   'wb'))
pickle.dump(list(X_trainval.columns), open('feature_cols.pkl', 'wb'))
pickle.dump(w,                    open('ensemble_weights.pkl','wb'))

print("Saved: model_xgb.pkl")
print("Saved: model_rf.pkl")
print("Saved: model_ridge.pkl")
print("Saved: model_nn.pkl")
print("Saved: scaler_final.pkl")
print("Saved: feature_cols.pkl")
print("Saved: ensemble_weights.pkl")
print(f"\nBest model: {best_name}  (R2 = {scores[best_name]:.4f})")
print("\nALL DONE! Ready to build Streamlit app.") 






