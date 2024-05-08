import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from pytz import utc
from timezonefinder import TimezoneFinder
from pytz import timezone


# Function to load the model and encoders
def load_model_and_encoders():
    with open('fraud_detection_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    encoders = {}
    for col in ['merchant', 'category', 'gender', 'city', 'state', 'job']:
        with open(f'{col}_encoder.pkl', 'rb') as encoder_file:
            encoders[col] = pickle.load(encoder_file)
    return model, encoders


# Function to calculate local time based on latitude and longitude
def get_local_time(dt, lat, lon):
    tf = TimezoneFinder()
    tz_str = tf.timezone_at(lat=lat, lng=lon)
    local_dt = dt.replace(tzinfo=utc).astimezone(timezone(tz_str))
    return local_dt


# Load the model and encoders
model, encoders = load_model_and_encoders()

# Title for the app
st.title('Fraud Detection System')
st.write("Please enter the transaction details:")

# Creating a form for user inputs
with st.form(key='transaction_form'):
    trans_dt = st.text_input("Transaction Date Time (yyyy-mm-dd hh:mm:ss)", "2020-06-21 12:14:00")
    cc_num = st.text_input("Credit Card Number", "2291163933867244")
    merchant = st.selectbox("Merchant", options=sorted(encoders['merchant'].classes_))
    category = st.selectbox("Category", options=sorted(encoders['category'].classes_))
    amt = st.number_input("Transaction Amount", min_value=0.01)
    gender = st.selectbox("Gender", options=sorted(encoders['gender'].classes_))
    city = st.selectbox("City", options=sorted(encoders['city'].classes_))
    state = st.selectbox("State", options=sorted(encoders['state'].classes_))
    lat = st.number_input("Latitude")
    long = st.number_input("Longitude")
    city_pop = st.number_input("City Population", min_value=1, format='%d')
    job = st.selectbox("Job", options=sorted(encoders['job'].classes_))
    dob = st.text_input("Date of Birth (yyyy-mm-dd)", "1980-01-01")
    merch_lat = st.number_input("Merchant Latitude")
    merch_long = st.number_input("Merchant Longitude")
    submit_button = st.form_submit_button("Predict")

if submit_button:
    # Convert transaction date time and dob to datetime objects
    dt = datetime.strptime(trans_dt, '%Y-%m-%d %H:%M:%S')
    dob_dt = datetime.strptime(dob, '%Y-%m-%d').date()
    age = (datetime.today().date() - dob_dt).days // 365

    # Encode categorical data using the loaded encoders
    merchant_encoded = encoders['merchant'].transform([merchant])[0]
    category_encoded = encoders['category'].transform([category])[0]
    gender_encoded = encoders['gender'].transform([gender])[0]
    city_encoded = encoders['city'].transform([city])[0]
    state_encoded = encoders['state'].transform([state])[0]
    job_encoded = encoders['job'].transform([job])[0]

    # Prepare the input dataframe
    input_data = pd.DataFrame([[
        dt, cc_num, merchant_encoded, category_encoded, amt, gender_encoded, city_encoded, state_encoded, lat, long,
        city_pop, job_encoded, age, merch_lat, merch_long
    ]], columns=['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 'gender', 'city', 'state', 'lat',
                 'long', 'city_pop', 'job', 'age', 'merch_lat', 'merch_long'])

    # Predict the outcome using the loaded model
    prediction = model.predict(input_data)
    result = 'Fraudulent' if prediction[0] == 1 else 'Legitimate'
    st.write(f'The transaction is predicted to be: **{result}**')
