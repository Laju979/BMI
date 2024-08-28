import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.header('BMI Prediction Created by Perfection')

datafile = pd.read_csv('Dataset of Diabetes .csv')

df = datafile[['Urea', 'Cr', 'HbA1c', 'Cholesterol', 'TG', 'HDL', 'LDL', 'VLD', 'BMI']]

f = df[['Urea', 'Cr', 'HbA1c', 'Cholesterol', 'TG', 'HDL', 'LDL', 'VLD']]
c = df[['BMI']]

feature_train, feature_test, target_train, target_test = train_test_split(f, c, test_size=0.2)

model = LinearRegression()
model.fit(feature_train, target_train)


st.sidebar.header('DIABETES CHECK')
Urea = st.sidebar.number_input('Urea', min_value=0)
Cr = st.sidebar.number_input('Cr', min_value=0)
HbA1c = st.sidebar.number_input('HbA1c', min_value=0)
Cholesterol = st.sidebar.number_input('Cholesterol', min_value=0)
TG = st.sidebar.number_input('TG', min_value=0)
HDL = st.sidebar.number_input('HDL', min_value=0)
LDL = st.sidebar.number_input('LDL', min_value=0)
VLD = st.sidebar.number_input('VLD', min_value=0)


columns = {'Urea': [Urea],
           'Cr': [Cr],
           'HbA1c': [HbA1c],
           'Cholesterol': [Cholesterol],
           'TG': [TG],
           'HDL': [HDL],
           'LDL': [LDL],
           'VLD': [VLD]}



input_features = pd.DataFrame(columns)
st.write('Personal Details:', input_features)


if st.button('My BMI'):
    prediction = model.predict(input_features)
    st.write(f'Your BMI based on what you have given us is: {prediction[0][0]:,.2f}')
