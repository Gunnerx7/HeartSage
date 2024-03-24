import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# HeartSage: Navigating Your Cardiovascular Future

This app predicts if a patient has heart disease.

Data obtained from Kaggle: [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
def user_input_features():
    age = st.sidebar.number_input('Enter your age: ')
    sex  = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    cp = st.sidebar.selectbox('Chest pain type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    tres = st.sidebar.number_input('Resting blood pressure: ')
    chol = st.sidebar.number_input('Serum cholestoral in mg/dl: ')
    fbs = st.sidebar.selectbox('Fasting blood sugar', ['Lower than 120 mg/dl', 'Greater than 120 mg/dl'])
    res = st.sidebar.selectbox('Resting electrocardiographic results', ['Normal', 'ST-T wave abnormality', 'Probable or definite left ventricular hypertrophy'])
    tha = st.sidebar.number_input('Maximum heart rate achieved: ')
    exa = st.sidebar.selectbox('Exercise induced angina', ['No', 'Yes'])
    old = st.sidebar.number_input('Oldpeak')
    slope = st.sidebar.selectbox('The slope of the peak exercise ST segment', ['Upsloping', 'Flat', 'Downsloping'])
    ca = st.sidebar.selectbox('Number of major vessels', ['0', '1', '2', '3'])
    thal = st.sidebar.selectbox('Thal', ['Normal', 'Fixed defect', 'Reversible defect'])

    data = {'age': age,
            'sex': 1 if sex == 'Female' else 0, 
            'cp': ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp),
            'trestbps': tres,
            'chol': chol,
            'fbs': 1 if fbs == 'Greater than 120 mg/dl' else 0,
            'restecg': ['Normal', 'ST-T wave abnormality', 'Probable or definite left ventricular hypertrophy'].index(res),
            'thalach': tha,
            'exang': 1 if exa == 'Yes' else 0,
            'oldpeak': old,
            'slope': ['Upsloping', 'Flat', 'Downsloping'].index(slope),
            'ca': ['0', '1', '2', '3'].index(ca),
            'thal': ['Normal', 'Fixed defect', 'Reversible defect'].index(thal)
           }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Reads in saved classification model
load_clf = pickle.load(open('Random_forest_model.pkl', 'rb'))

# Apply feature transformation to match the format used during training
df = pd.read_csv('heart.csv')
df = df.drop(columns=['target'])

# Combine user input features with entire dataset for the encoding phase
combined_df = pd.concat([input_df, df], axis=0)

# Encoding of categorical features
combined_df = pd.get_dummies(combined_df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Select only the first row (user input data)
input_df_encoded = combined_df.iloc[:1]

st.write(input_df_encoded)

# Apply model to make predictions
prediction = load_clf.predict(input_df_encoded)
prediction_proba = load_clf.predict_proba(input_df_encoded)

st.subheader('Prediction')
if prediction[0] == 1:
    st.write("The patient is predicted to have heart disease.")
    st.subheader('Treatment Measures')
    st.write("""
    - Seek immediate medical attention.
    - Follow the treatment plan prescribed by your healthcare provider.
    - Make lifestyle changes such as adopting a healthy diet and regular exercise.
    - Take medications as prescribed.
    - Attend follow-up appointments with your healthcare provider.
    """)
else:
    st.write("The patient is predicted not to have heart disease.")
    st.subheader('Precautions')
    st.write("""
    - Maintain a healthy lifestyle with regular exercise and a balanced diet.
    - Avoid smoking and excessive alcohol consumption.
    - Manage stress through relaxation techniques such as meditation or yoga.
    - Monitor blood pressure, cholesterol levels, and other risk factors regularly.
    - Consult a healthcare provider for regular check-ups and preventive care.
    """)
st.subheader('Prediction Probability')
st.write(prediction_proba)
