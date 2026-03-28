# Regression-model
Predicts house prices using Ridge Regression | Ames Housing Dataset | Streamlit App



# House Price Predictor

A machine learning web app that predicts house prices based on the Ames Housing dataset.

## Live Demo
[Click here to try the app](your streamlit link here)

## Project Overview
- Dataset: Ames Housing Dataset (2930 houses, 80 features)
- Problem: Predict house sale price (Regression)
- Best Model: Ridge Regression (alpha=10)
- Final CV R2 Score: 0.8303

## Steps Followed
- Data Cleaning (missing values, encoding)
- EDA (correlation analysis, feature selection)
- Outlier Removal (IQR method)
- Feature Selection (|corr| >= 0.5 → 15 features)
- Model Training (Linear Regression + Ridge)
- Cross Validation (cv=5)
- Deployed as web app using Streamlit

## Tech Stack
- Python
- Pandas, Numpy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit

## How to Run Locally
pip install -r requirements.txt
streamlit run app.py

## Results
| Model                | CV R2  | Test R2 |
|----------------------|--------|---------|
| Linear Regression    | 0.7946 | 0.7421  |
| Ridge (alpha=10)     | 0.8303 | 0.7063  |
