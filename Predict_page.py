import streamlit as st
import pickle
import numpy as np

def load_model():
    with open(r'C:\Users\nares\OneDrive\Documents\PythonPrograms\Diabetes_Prediction\saved_steps.pkl','rb') as file:
       data = pickle.load(file)
    return data

data = load_model()

lg = data['model']
le_gender = data['le_gender']
le_hypertension = data['le_hypertension']
le_heart_disease = data['le_heart_disease']
le_smoking_history = data['le_smoking_history']

def show_predict_page():
    st.title("Diabetes Prediction Model using Logistic Regression")
    st.write("""### Enter Inputs below""")

show_predict_page()

gender = (
        "Male",
        "Female",
        "Other",
    )

hypertension = (
        'Yes',
        'No',
    )

    
heart_disease = (
        'Yes',
        'No',
    )

smoking_history = (
        'never',
        'former',
        'current',
    )


gender = st.selectbox('Gender',gender)
age = st.slider('Select Age',0,80,3)
hypertension = st.selectbox('Have Hypertension?',hypertension)
heart_disease = st.selectbox('Have any Heart Dieses?',heart_disease)
smoking_history = st.selectbox('How offen do you smoke?',smoking_history)
bmi = st.slider('Select BMI(Body Mass Index)',15.0,50.0,20.0)
HbA1c = st.slider('Select HbA1c (Hemoglobin A1c) level', 3.5, 9.0, 4.5)
blood_glucose_level = st.slider('Select Your Blooad glucose level',80,300,150)

OK = st.button("Predict Diabetes")
if OK:
    X = np.array([[gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c,blood_glucose_level]])
    X[:, 0]= le_gender.fit_transform(X[:,0])
    X[:, 2]= le_hypertension.fit_transform(X[:,2])
    X[:, 3]= le_heart_disease.fit_transform(X[:,3])
    X[:, 4]= le_smoking_history.fit_transform(X[:,4])
    X = X.astype(float)

    prediction = lg.predict(X)

    if 0 in prediction:
       st.title("You have NO Diabetes")
    else:
       st.title("You have Diabetes")



