import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load encoders and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scalar.pkl', 'rb') as file:
    scalar = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Build input data from user inputs
input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

# Encode gender
input_data['Gender'] = label_encoder_gender.transform([input_data['Gender']])[0]

# One-hot encode geography
geo_encoded = label_encoder_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

# Build dataframe
input_df = pd.DataFrame([input_data]).drop(columns=['Geography'])
input_df = pd.concat([input_df, geo_encoded_df], axis=1)

# Scale and predict
input_scaled = scalar.transform(input_df)
input_scaled = input_scaled[:, :-1]

prediction = model.predict(input_scaled)
churn_probability = prediction[0][0]

# Display result
st.write(f"### Churn Probability: {churn_probability:.2f}")
if churn_probability > 0.5:
    st.error("⚠️ Customer is likely to churn!")
else:
    st.success("✅ Customer is not likely to churn.")
    st.balloons()  