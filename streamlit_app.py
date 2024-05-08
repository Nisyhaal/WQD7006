import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import math
from pytz import utc
from timezonefinder import TimezoneFinder
from pytz import timezone

# Function to load the model and encoders
def load_model_and_encoders():
    with open('best_trained_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    encoders = {}
    for col in ['category', 'gender', 'state', 'user_time_of_day', 'merch_time_of_day', 'card_issuer', 'mii']:
        with open(f'{col}_encoder.pkl', 'rb') as encoder_file:
            encoders[col] = pickle.load(encoder_file)
    return model, encoders

# Haversine distance calculation
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    delta_lat, delta_lon = lat2_rad - lat1_rad, lon2_rad - lon1_rad
    a = np.sin(delta_lat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Card issuer and MII identification functions
def get_card_issuer(number):
    number = str(number)
    first_digit = number[0]
    return ('Visa' if first_digit == '4' else 'Mastercard' if first_digit in ['2', '5'] else
            'American Express' if first_digit == '3' else 'Discover' if first_digit == '6' else 'Unknown')

def get_mii(number):
    number = str(number)
    first_digit = number[0]
    if first_digit == '1':
        return 'Air travel and financial services'
    elif first_digit in ['2', '3', '4', '5', '6']:
        return 'Credit card networks'
    elif first_digit == '7':
        return 'Petroleum'
    elif first_digit == '8':
        return 'Healthcare and telecommunications'
    elif first_digit == '9':
        return 'Government and "other" category'
    else:
        return 'Unknown'

# Timezone and time of day calculation functions
def get_local_time(lat, lon, dt):
    tf = TimezoneFinder()
    tz_str = tf.timezone_at(lat=lat, lng=lon)
    if tz_str:
        local_time = dt.replace(tzinfo=utc).astimezone(timezone(tz_str))
        return local_time
    return None

def time_of_day(local_time):
    if not local_time:
        return "Timezone Not Found"
    hour = local_time.hour
    return ('Morning' if 5 <= hour < 12 else 'Afternoon' if 12 <= hour < 17 else
            'Evening' if 17 <= hour < 21 else 'Night')

model, encoders = load_model_and_encoders()

st.title('Fraud Detection System')
st.write("Please enter the transaction details:")

with st.form(key='transaction_form'):
    trans_dt = st.text_input("Transaction Date Time (yyyy-mm-dd hh:mm:ss)", "2020-06-21 12:14:00")
    cc_num = st.text_input("Credit Card Number", "2291163933867244")
    category = st.selectbox("Category", options=['personal_care', 'health_fitness', 'misc_pos', 'travel', 'kids_pets', 'shopping_pos', 'food_dining', 'home', 'entertainment', 'shopping_net', 'misc_net', 'grocery_pos', 'gas_transport', 'grocery_net'])
    amt = st.number_input("Transaction Amount", min_value=0.01)
    gender = st.selectbox("Gender", options=['M', 'F'])
    state = st.selectbox("State", options=['SC', 'UT', 'NY', 'FL', 'MI', 'CA', 'SD', 'PA', 'TX', 'KY', 'WY', 'AL', 'LA', 'GA', 'CO', 'OH', 'WI', 'VT', 'AR', 'NJ', 'IA', 'MD', 'MS', 'KS', 'IL', 'MO', 'ME', 'TN', 'DC', 'AZ', 'MT', 'MN', 'OK', 'WA', 'WV', 'NM', 'MA', 'NE', 'VA', 'ID', 'OR', 'IN', 'NC', 'NH', 'ND', 'CT', 'NV', 'HI', 'RI', 'AK'])
    lat = st.number_input("Latitude")
    long = st.number_input("Longitude")
    city_pop = st.number_input("City Population", min_value=1)
    age = st.number_input("Age", min_value=1)
    merch_lat = st.number_input("Merchant Latitude")
    merch_long = st.number_input("Merchant Longitude")
    submit_button = st.form_submit_button("Predict")

if submit_button:
    dt = datetime.strptime(trans_dt, '%Y-%m-%d %H:%M:%S')
    distance = haversine_distance(lat, long, merch_lat, merch_long)
    user_local_time = get_local_time(lat, long, dt)
    merch_local_time = get_local_time(merch_lat, merch_long, dt)
    user_time_of_day = time_of_day(user_local_time)
    merch_time_of_day = time_of_day(merch_local_time)
    card_issuer = get_card_issuer(cc_num)
    mii = get_mii(cc_num)

    # Encode categorical data using the loaded encoders
    category_encoded = encoders['category'].transform([category])[0]
    gender_encoded = encoders['gender'].transform([gender])[0]
    state_encoded = encoders['state'].transform([state])[0]
    user_time_of_day_encoded = encoders['user_time_of_day'].transform([user_time_of_day])[0]
    merch_time_of_day_encoded = encoders['merch_time_of_day'].transform([merch_time_of_day])[0]
    card_issuer_encoded = encoders['card_issuer'].transform([card_issuer])[0]
    mii_encoded = encoders['mii'].transform([mii])[0]

    # Prepare the input dataframe
    input_data = pd.DataFrame([[
        category_encoded, amt, gender_encoded, state_encoded, city_pop, age, user_time_of_day_encoded, merch_time_of_day_encoded, card_issuer_encoded, mii_encoded, distance
    ]], columns=['category', 'amt', 'gender', 'state', 'city_pop', 'age', 'user_time_of_day', 'merch_time_of_day', 'card_issuer', 'mii', 'distance'])

    # Predict the outcome using the loaded model
    prediction = model.predict(input_data)
    result = 'Fraudulent' if prediction[0] == 1 else 'Legitimate'
    st.write(f'The transaction is predicted to be: **{result}**')
