import pickle
import numpy as np
import streamlit as st
import os
import string
model = pickle.load(open('heartfailuremodel.pkl', 'rb'))

def main():
    st.title('Heart Failure Prediction Model')

    #input variables
    Age = st.text_input('Input your age')
    Sex = st.text_input('Input your sex')
    ChestPainType = st.text_input('What is the Type of chest pains you are experiencing?')
    RestingBP = st.text_input('What is your BP rate while resting?')
    Cholesterol = st.text_input('What is your cholesterol level?')
    FastingBS = st.text_input('What is your BS when fasting?')
    RestingECG = st.text_input('What is your ECG when resting?')
    MaxHR = st.text_input('What is your max HR?')
    ExerciseAngina = st.text_input('Your Exercise Angina')
    Oldpeak = st.text_input('What is your Oldpeak')
    ST_Slope = st.text_input('Input your ST_Slope')
    makeprediction = ""


    #prediction code
    if st.button('Predict'):
        makeprediction = model.predict([[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]])
        output = round(makeprediction[0],2)
        st.success('The chance of getting heart failure {}'.format(output))

if __name__=="__main__":
    main()        

