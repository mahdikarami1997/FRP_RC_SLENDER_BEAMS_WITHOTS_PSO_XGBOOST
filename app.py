import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np

# --- 1. Train the Model on the Fly ---
# We use @st.cache_resource so it only trains once when the app starts, not every time you click.
@st.cache_resource
def load_and_train_model():
    # Load data
    try:
        data = pd.read_csv('Data_Cleaned.csv')
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        # Train model (using parameters close to your best results)
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

model = load_and_train_model()

# --- 2. The Website Interface ---
st.title("Concrete Shear Strength Predictor")
st.write("Enter the parameters below to predict the shear strength (Vcf).")

# Create input fields side-by-side
col1, col2 = st.columns(2)

with col1:
    bw = st.number_input("Beam Width (bw)(mm)", value=0)
    a  = st.number_input("Shear Span (a)(mm)", value=0)
    d  = st.number_input("Effective Depth (d)(mm)", value=0)
    fc = st.number_input("Concrete Strength (f'c)(MPa)", value=0)

with col2:
    p  = st.number_input("Reinforcement Ratio (p)(%)", value=0)
    Ef = st.number_input("FRP Modulus (Ef)(GPa)", value=40)
    Ec = st.number_input("Concrete Modulus (Ec)(GPa)", value=0)

# --- 3. Prediction ---
if st.button("Calculate Strength"):
    if model:
        # Prepare input
        input_data = np.array([[bw, a, d, fc, p, Ef, Ec]])
        
        # Predict
        prediction = model.predict(input_data)[0]
        
        # Show result
        st.success(f"Predicted Shear Strength (Vcf)(KN): {prediction:.4f}")
    else:
        st.error("Model could not be trained. Please check the dataset file.")