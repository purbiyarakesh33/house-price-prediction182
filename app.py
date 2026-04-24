import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="",
    layout="wide"
)

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    xgb        = pickle.load(open('model_xgb.pkl',       'rb'))
    rf         = pickle.load(open('model_rf.pkl',        'rb'))
    ridge      = pickle.load(open('model_ridge.pkl',     'rb'))
    scaler     = pickle.load(open('scaler_final.pkl',    'rb'))
    feat_cols  = pickle.load(open('feature_cols.pkl',    'rb'))
    ens_w      = pickle.load(open('ensemble_weights.pkl','rb'))
    return xgb, rf, ridge, scaler, feat_cols, ens_w

xgb_model, rf_model, ridge_model, scaler, feature_cols, ens_weights = load_models()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title(" House Price Predictor")
st.markdown("##### Ames Housing Dataset — Ensemble Model (R² = 0.8441)")
st.markdown("---")

# ── SIDEBAR INFO ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header(" Model Info")
    st.markdown("""
    **Models Used:**
    - XGBoost (R² = 0.8410)
    - Random Forest (R² = 0.8295)
    - Ridge Regression (R² = 0.8266)

    **Ensemble R² = 0.8441**
    
    **Top Predictors:**
    1. Quality × Total SF
    2. Quality × Living Area
    3. Overall Quality
    4. Total Square Footage
    5. Garage Cars
    """)
    st.markdown("---")
    st.caption("Built by: Rakesh")
    st.caption("Dataset: Ames Housing")

# ── INPUT FORM ────────────────────────────────────────────────────────────────
st.subheader("Enter House Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Size & Area**")
    gr_liv_area   = st.number_input("Living Area (sqft)",      300,  6000, 1500)
    total_bsmt_sf = st.number_input("Total Basement SF",         0,  3000, 1000)
    first_flr_sf  = st.number_input("1st Floor SF",            500,  5000, 1200)
    garage_area   = st.number_input("Garage Area (sqft)",        0,  1500,  500)
    mas_vnr_area  = st.number_input("Masonry Veneer Area (sqft)",0,  1500,    0)

with col2:
    st.markdown("**Quality & Condition**")
    overall_qual    = st.slider("Overall Quality (1-10)", 1, 10, 5)
    exter_qual_ta   = st.selectbox("Exterior Quality — Average?",  ["No", "Yes"])
    bsmt_qual_ex    = st.selectbox("Basement Quality — Excellent?",["No", "Yes"])
    kitchen_qual_ex = st.selectbox("Kitchen Quality — Excellent?", ["No", "Yes"])
    kitchen_qual_ta = st.selectbox("Kitchen Quality — Average?",   ["No", "Yes"])
    foundation_pconc= st.selectbox("Concrete Foundation?",         ["No", "Yes"])

with col3:
    st.markdown("**Age & Garage**")
    year_built   = st.number_input("Year Built",       1900, 2023, 2000)
    year_remod   = st.number_input("Year Remodelled",  1900, 2023, 2000)
    garage_cars  = st.slider("Garage Cars",  0, 4, 2)
    full_bath    = st.slider("Full Bathrooms", 0, 4, 2)

st.markdown("---")

# ── PREDICT BUTTON ────────────────────────────────────────────────────────────
if st.button("Predict Price", use_container_width=True):

    # Convert Yes/No to 1/0
    def yn(val): return 1 if val == "Yes" else 0

    # ── ENGINEER FEATURES (same as training) ──────────────────────────────────
    TotalSF          = total_bsmt_sf + first_flr_sf + gr_liv_area
    BsmtRatio        = total_bsmt_sf / (first_flr_sf + 1)
    HouseAge         = 2010 - year_built
    RemodAge         = 2010 - year_remod
    Was_Remodeled    = 1 if year_remod != year_built else 0
    HasGarage        = 1 if garage_area > 0 else 0
    GarageEfficiency = garage_area / (garage_cars + 1)
    HasMasVnr        = 1 if mas_vnr_area > 0 else 0
    Qual_x_Area      = overall_qual * gr_liv_area
    Qual_x_TotalSF   = overall_qual * TotalSF
    KitchenScore     = (yn(kitchen_qual_ex) * 2 +
                       (1 - yn(kitchen_qual_ta)) * (1 - yn(kitchen_qual_ex)))

    # ── BUILD INPUT ROW ───────────────────────────────────────────────────────
    input_dict = {
        'Overall Qual'     : overall_qual,
        'Year Built'       : year_built,
        'Year Remod/Add'   : year_remod,
        'Mas Vnr Area'     : mas_vnr_area,
        'Total Bsmt SF'    : total_bsmt_sf,
        '1st Flr SF'       : first_flr_sf,
        'Gr Liv Area'      : gr_liv_area,
        'Full Bath'        : full_bath,
        'Garage Cars'      : garage_cars,
        'Garage Area'      : garage_area,
        'Exter Qual_TA'    : yn(exter_qual_ta),
        'Foundation_PConc' : yn(foundation_pconc),
        'Bsmt Qual_Ex'     : yn(bsmt_qual_ex),
        'Kitchen Qual_Ex'  : yn(kitchen_qual_ex),
        'Kitchen Qual_TA'  : yn(kitchen_qual_ta),
        'TotalSF'          : TotalSF,
        'BsmtRatio'        : BsmtRatio,
        'HouseAge'         : HouseAge,
        'RemodAge'         : RemodAge,
        'Was_Remodeled'    : Was_Remodeled,
        'HasGarage'        : HasGarage,
        'GarageEfficiency' : GarageEfficiency,
        'HasMasVnr'        : HasMasVnr,
        'Qual_x_Area'      : Qual_x_Area,
        'Qual_x_TotalSF'   : Qual_x_TotalSF,
        'KitchenScore'     : KitchenScore,
    }

    # Build dataframe with all training columns, fill missing with 0
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    # Scale for ridge
    input_scaled = scaler.transform(input_df)

    # ── PREDICTIONS ───────────────────────────────────────────────────────────
    xgb_pred   = np.expm1(xgb_model.predict(input_df)[0])
    rf_pred    = np.expm1(rf_model.predict(input_df)[0])
    ridge_pred = np.expm1(ridge_model.predict(input_scaled)[0])

    # Ensemble using saved optimal weights
    w = ens_weights
    log_ensemble = (w[0] * np.log1p(ridge_pred) +
                    w[1] * np.log1p(rf_pred) +
                    w[2] * np.log1p(xgb_pred))
    ensemble_pred = np.expm1(log_ensemble)

    # Confidence range ± 10%
    low  = ensemble_pred * 0.90
    high = ensemble_pred * 1.10

    # ── DISPLAY RESULTS ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(" Prediction Results")

    res1, res2, res3 = st.columns(3)
    with res1:
        st.metric("XGBoost",       f"${xgb_pred:,.0f}")
    with res2:
        st.metric("Random Forest", f"${rf_pred:,.0f}")
    with res3:
        st.metric("Ridge",         f"${ridge_pred:,.0f}")

    st.markdown("###")
    st.success(f"###  Ensemble Predicted Price: **${ensemble_pred:,.0f}**")
    st.info(f"**Confidence Range:** ${low:,.0f}  —  ${high:,.0f}")

    # ── FEATURE SUMMARY ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(" Your House Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total SF",        f"{TotalSF:,.0f} sqft")
    s2.metric("Overall Quality", f"{overall_qual} / 10")
    s3.metric("House Age",       f"{HouseAge} years")
    s4.metric("Qual x TotalSF",  f"{Qual_x_TotalSF:,.0f}")
