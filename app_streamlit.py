import streamlit as st
import pandas as pd
import joblib

best_rf_model = joblib.load("best_rf_model.joblib")

st.set_page_config(
	page_title="Bank Marketing Campaign Classification",
	page_icon=":bank:"
)

st.title(":bank: Bank Marketing Campaign Classification")
st.markdown("Klasifikasi Bank Marketing untuk memprediksi nasabah akan berlangganan deposito berjangka.")

age = st.number_input("Age", min_value=18, max_value=100, step=1)
job = st.selectbox("Job", ['housemaid', 'entrepreneur', 'self-employed', 'admin', 'unknown'])
marital = st.selectbox("Marital", ['married', 'single', 'divorced'])
education = st.selectbox("Education", ['primary','secondary','tertiary','unknown'])
balance = st.slider("Balance", 25, 33651, 9861)
housing = st.pills("Housing", ['yes', 'no'], default='yes')
loan = st.pills("Loan", ['yes', 'no'], default='no')

if st.button("Prediksi", type="primary"):
	data_baru = pd.DataFrame([[age,job,marital,education,balance,housing,loan]], columns=['age','job','marital','education','balance','housing','loan'])

	prediksi = best_rf_model.predict(data_baru)[0]
	presentase = max(best_rf_model.predict_proba(data_baru)[0])
	st.success(f"Prediksi apakah nasabah akan berlangganan : {prediksi}")
	st.error(f"Tingkat keyakinan : {presentase*100:.2f}%")
	st.balloons()

st.divider()

st.caption("Dibuat untuk seleksi LKS - classification")
