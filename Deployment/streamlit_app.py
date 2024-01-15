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

training_data = pd.read_csv('heart.csv')

# Fit encoders with training data
binary_var = ['Sex', 'FastingBS', 'ExerciseAngina', 'RestingECG']
enc_oe.fit(training_data[binary_var])

multi_categ = ['ChestPainType', 'ST_Slope']
enc_ohe.fit(training_data[multi_categ])

# Fit StandardScaler with numeric variables
numeric_vars = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler.fit(training_data[numeric_vars])

def make_prediction_knn_preprocessed(model, input_data, enc_oe, enc_ohe, scaler, training_data): 
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Define the columns used during training
    training_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

    # Transform categorical columns using OrdinalEncoder
    binary_var = ['Sex', 'FastingBS', 'ExerciseAngina', 'RestingECG']
    
    enc_oe.fit(input_df[['Sex', 'FastingBS', 'ExerciseAngina', 'RestingECG']])
    
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
    # st.title('Heart Disease Prediction')
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction Model", "Learn about Heart Failure"])

    if page == "Prediction Model":
        # Home page content
        st.title('Heart Failure Prediction Model')

        # Input widgets for user input
        Age = st.number_input('Enter your age', min_value=0, max_value=120, value=25)
        Sex = st.selectbox('Select your gender', ['Male', 'Female'])
        
        # Explanation for Chest Pain Types
        chest_pain_explanations = {
            'TA': 'Typical Angina - Chest pain or discomfort with a typical pattern, often described as pressure or squeezing.',
            'ATA': 'Atypical Angina - Chest pain that does not fit the typical patterns of classic angina.',
            'NAP': 'Non-Anginal Pain - Chest discomfort not related to the heart; may be musculoskeletal or gastrointestinal.',
            'ASY': 'Asymptomatic - No chest pain or discomfort.'
        }
        
        ChestPainType = st.selectbox('Select your chest pain type', list(chest_pain_explanations.keys()))
        st.write(chest_pain_explanations.get(ChestPainType, ''))
        
        RestingBP = st.number_input('Enter your resting blood pressure (mm Hg)', min_value=0, value=120)
        Cholesterol = st.number_input('Enter your cholesterol level (mg/dl)', min_value=0, value=200)
        FastingBS = st.selectbox('Is your fasting blood sugar > 120 mg/dl?', ['Yes', 'No'])
        RestingECG = st.selectbox('What is your resting electrocardiographic result?', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
        MaxHR = st.number_input('Enter your maximum heart rate', min_value=0, value=150)
        ExerciseAngina = st.selectbox('Do you experience exercise-induced angina?', ['Yes', 'No'])
        Oldpeak = st.number_input('Enter your ST depression induced by exercise relative to rest', value=0.0)
        ST_Slope = st.selectbox('Select your slope of the peak exercise ST segment', ['Up', 'Flat', 'Down'])

    
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
            predicted_heart_disease_knn_preprocessed = make_prediction_knn_preprocessed(classifier_KNN, input_data, enc_oe, enc_ohe, scaler, training_data)
            # st.success('The predicted result is {}'.format(predicted_heart_disease_knn_preprocessed))
            # Print result based on predicted value
            if predicted_heart_disease_knn_preprocessed == 1:
                st.warning('There is a possibility of heart failure.')
        else:
            st.success('There is no possibility of heart failure.')
            
    elif page == "Learn about Heart Failure":
        # Content for the "Learn about Heart Failure" page
        st.title("Learn about Heart Failure")
    
        # Definition of Heart Failure
        st.header("Definition of Heart Failure")
        st.write("Heart failure is a condition where the heart is unable to pump blood efficiently, leading to inadequate oxygen supply to the body's tissues.")
    
        # Causes of Heart Failure
        st.header("Causes of Heart Failure")
        st.write("Common causes include coronary artery disease, high blood pressure, and heart attack.")
    
        # Ways to Deal with Heart Failure
        st.header("Ways to Deal with Heart Failure")
        st.write("1. Medications\n2. Lifestyle changes\n3. Dietary modifications\n4. Exercise\n5. Regular checkups")
    
        # Foods to Eat or Avoid
        st.header("Foods to Eat or Avoid")
        st.write("Include heart-healthy foods such as fruits, vegetables, and whole grains. Avoid excessive salt and saturated fats.")
    
        # Exercises to Do
        st.header("Exercises to Do")
        st.write("Engage in aerobic exercises like walking, swimming, and cycling. Consult your healthcare provider before starting a new exercise routine.")
    
        # Checkup Recommendations
        st.header("Checkup Recommendations")
        st.write("If at risk for heart failure, it is advisable to have regular checkups. Consult your healthcare provider for personalized recommendations.")
    
        # Additional Information and Resources
        st.header("Additional Information and Resources")
    
        # Provide links to educational videos on YouTube
        st.subheader("Educational Videos on YouTube:")
        st.markdown("[Understanding Congestive Heart Failure - Alila Medical Media](https://www.youtube.com/watch?v=b3OHSA7Tz7U&pp=ygUZaGVhcnQgZmFpbHVyZSBkb2N1bWVudGFyeQ%3D%3D)")
        st.markdown("[Living with Heart Failure - Guide for Your Daily Activities](https://www.youtube.com/watch?v=4ZI30tBFLUw&pp=ygUlbGl2aW5nIHdpdGggaGVhcnQgZmFpbHVyZSBkb2N1bWVudGFyeQ%3D%3D)")
    
        # Other sources of information
        st.subheader("Other Sources of Information:")
        st.markdown("[Heart failure - Symptoms and causes (Mayo Clinic)](https://www.mayoclinic.org/diseases-conditions/heart-failure/symptoms-causes/syc-20373142)")
        st.markdown("[Living with Heart Failure (HF) - Recovery & Management (British Heart Foundation)](https://www.bhf.org.uk/informationsupport/support/practical-support/living-with-heart-failure)")     

if __name__ == "__main__":
    main()
