import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('transaction_amount_predictor.pkl')

# Initialize StandardScaler
scaler = StandardScaler()

st.title('Transaction Amount Predictor')

# User input
account_number = st.number_input('Enter Account Number:', min_value=1000, max_value=9999, value=1000)
transaction_type = st.selectbox('Transaction Type:', ['Deposit', 'Withdrawal', 'ATM'])
balance = st.number_input('Enter Account Balance:', min_value=100, max_value=10000, value=100)

# Encode transaction type (This encoding should match with what you've done during training)
transaction_type_encoded = 0 if transaction_type == 'Deposit' else 1 if transaction_type == 'Withdrawal' else 2

# Prepare the feature vector
features = np.array([[account_number, transaction_type_encoded, balance]])

# Scale the features
features_scaled = scaler.fit_transform(features)

# Make prediction
if st.button('Predict Transaction Amount'):
    prediction = model.predict(features_scaled)
    st.write(f'Predicted Transaction Amount: {prediction[0]}')
