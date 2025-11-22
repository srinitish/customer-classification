import joblib
import streamlit as st
import pandas as pd

scale = joblib.load('scaler.plk')
model = joblib.load('model.plk')

st.title("Customer Segmentation using K-Means")
st.markdown("Enter new customer details to predict their segment.")

cluster_labels = {
    4: "Budget Customers",
    0: "Standard Shoppers",
    1: "Target Customers (High Income & Spending)",
    3: "Potential Customers (High Income, Low Spending)",
    2: "Low Income, High Spending"
}


income = st.number_input("Annual Income (k$)", min_value=10, max_value=150, value=50)
spending = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

if st.button("Predict Cluster"):
    new_data = pd.DataFrame([[income, spending]], columns=['Annual Income (k$)', 'Spending Score (1-100)'])
    new_scaled = scale.transform(new_data)
    cluster = model.predict(new_scaled)[0]
    st.success(f"Predicted Cluster: {cluster} - {cluster_labels.get(cluster, 'Unknown')}")