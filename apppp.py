import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the hybrid model (XGBoost + LSTM combined)
@st.cache_resource
def load_model():
    model = joblib.load("scaler.pkl")  # Update with your actual model file
    return model

model = load_model()

# Streamlit UI
st.set_page_config(page_title="Hybrid XGB + LSTM Solar Prediction", layout="wide")
st.title("🌞 Solar Energy Prediction (Hybrid XGBoost + LSTM)")

# User Input Method Selection
input_method = st.radio("Choose Input Method", ["Manual Input", "Upload CSV"])

if input_method == "Manual Input":
    st.subheader("📌 Enter Feature Values for Prediction")

    # User Input Fields (Exact Features from the Model)
    ambient_temp = st.number_input("🌡️ AMBIENT_TEMPERATURE", value=29.91)
    module_temp = st.number_input("🔥 MODULE_TEMPERATURE", value=45.68)
    irradiation = st.number_input("☀️ IRRADIATION", value=0.71)
    daily_yield = st.number_input("📈 DAILY_YIELD", value=2176.6)
    month = st.number_input("📅 MONTH", min_value=1, max_value=12, value=5)
    day = st.number_input("📆 DAY", min_value=1, max_value=31, value=29)
    hour = st.number_input("⏳ HOUR", min_value=0, max_value=23, value=10)

    # Convert input to DataFrame
    input_data = pd.DataFrame([[ambient_temp, module_temp, irradiation, daily_yield, month, day, hour]], 
                              columns=['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 
                                       'DAILY_YIELD', 'MONTH', 'DAY', 'HOUR'])

    if st.button("🔮 Predict AC Power"):
        prediction = model.predict(input_data)
        st.success(f"⚡ **Predicted AC Power:** {prediction[0]:.2f} kW")

elif input_method == "Upload CSV":
    st.subheader("📂 Upload CSV File with Required Features")
    
    uploaded_file = st.file_uploader("📤 Upload CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Sample:", df.head())

        # Ensure correct feature selection
        required_columns = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 
                            'DAILY_YIELD', 'MONTH', 'DAY', 'HOUR']
        
        if not all(col in df.columns for col in required_columns):
            st.error(f"❌ CSV must contain columns: {', '.join(required_columns)}")
        else:
            # Model Predictions
            predictions = model.predict(df[required_columns])

            # Add Predictions to DataFrame
            df['Predicted AC Power'] = predictions
            st.write("✅ Predictions:", df[['Predicted AC Power']])

            # Plot Results
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df.index, df['Predicted AC Power'], label="Predicted AC Power", color="orange")
            ax.set_xlabel("Time")
            ax.set_ylabel("AC Power (kW)")
            ax.set_title("Predicted AC Power Over Time")
            ax.legend()
            st.pyplot(fig)

# Footer
st.markdown("🚀 Built with **Streamlit, XGBoost, LSTM**")
