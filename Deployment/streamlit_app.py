import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

with open('heartfailuremodel.pkl', 'rb') as model_file:
    classifier_KNN = pickle.load(model_file)

# Define the encoders and scaler
enc_oe = OrdinalEncoder()
enc_ohe = OneHotEncoder()
scaler = StandardScaler()

def make_prediction_knn_preprocessed(model, input_data, enc_oe, enc_ohe, scaler): 
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Define the columns used during training
    training_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

    # Transform categorical columns using OrdinalEncoder
    binary_var = ['Sex', 'FastingBS', 'ExerciseAngina', 'RestingECG']
    enc_oe.fit(input_df[binary_var])
    input_df[binary_var] = enc_oe.transform(input_df[binary_var])

    # Transform 'ChestPainType' and 'ST_Slope' using OneHotEncoder
    multi_categ = ['ChestPainType', 'ST_Slope']
    transformed = pd.DataFrame(enc_ohe.transform(input_df[multi_categ]).toarray(), columns=enc_ohe.get_feature_names_out(multi_categ))
    input_df = pd.concat([input_df, transformed], axis=1).drop(multi_categ, axis=1)

    # Define the numeric variables
    numeric_vars = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

    # Scale the numeric variables
    input_df[numeric_vars] = scaler.transform(input_df[numeric_vars])

    # Make a prediction using the preprocessed input data and the KNN classifier
    prediction = model.predict(input_df)

    return prediction

def main():
    st.title('Heart Disease Prediction')

    # Input widgets for user input
    Age = st.number_input('Enter your age', min_value=0, max_value=120, value=25)
    Sex = st.selectbox('Select your gender', ['Male', 'Female'])
    ChestPainType = st.selectbox('Select your chest pain type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    RestingBP = st.number_input('Enter your resting blood pressure (mm Hg)', min_value=0, value=120)
    Cholesterol = st.number_input('Enter your cholesterol level (mg/dl)', min_value=0, value=200)
    FastingBS = st.selectbox('Is your fasting blood sugar > 120 mg/dl?', ['Yes', 'No'])
    RestingECG = st.selectbox('What is your resting electrocardiographic results?', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
    MaxHR = st.number_input('Enter your maximum heart rate', min_value=0, value=150)
    ExerciseAngina = st.selectbox('Do you experience exercise-induced angina?', ['Yes', 'No'])
    Oldpeak = st.number_input('Enter your ST depression induced by exercise relative to rest', value=0.0)
    ST_Slope = st.selectbox('Select your slope of the peak exercise ST segment', ['Upsloping', 'Flat', 'Downsloping'])

    makeprediction = ""

    # Prediction code
    if st.button('Predict'):
        # Preprocess the input data
        input_data = {
            'Age': Age,
            'Sex': Sex,
            'ChestPainType': ChestPainType,
            'RestingBP': RestingBP,
            'Cholesterol': Cholesterol,
            'FastingBS': FastingBS,
            'RestingECG': RestingECG,
            'MaxHR': MaxHR,
            'ExerciseAngina': ExerciseAngina,
            'Oldpeak': Oldpeak,
            'ST_Slope': ST_Slope
        }

        # Make predictions
        predicted_heart_disease_knn_preprocessed = make_prediction_knn_preprocessed(classifier_KNN, input_data, enc_oe, enc_ohe, scaler)
        st.success('The predicted result is {}'.format(predicted_heart_disease_knn_preprocessed))

if __name__ == "__main__":
    main()
