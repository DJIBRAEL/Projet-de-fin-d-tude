# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:31:38 2024

@author: USER
"""

import numpy as np
import pickle 
import streamlit as st


loaded_model=pickle.load(open('C:/Users/USER/Desktop/Deploying Machine learning Model/trained_model.sav','rb'))

# create function
def diabetes_prediction(input_data):
    
   
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if(prediction[0]==0):
        return"la personne n'est pas diabétique"
    else:
        return "la personne est diabétique"
    
    
def main():
    # giving a title 
    st.title("La prediction de diabete application web")
# getting the input data from users    
    Pregnancies= st.text_input("Nombre de grosesse")
    Glucose=st.text_input("le degre de glcucose")
    BloodPressure=st.text_input("valeur de la pression artérielle")
    SkinThickness=st.text_input("épaisseur de la peau")
    Insulin=st.text_input("L'insuline")
    BMI=st.text_input("L'indice de masse corporelle (BMC)")
    DiabetesPedigreeFunction=st.text_input("fonction de pedigree du diabète")
    Age=st.text_input("Age")
    
    # code for prediction 
    diagnosis=''
    
    # creating a button for prediction 
    if st.button("Diabete Test Resultat"):
        diagnosis= diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        st.success(diagnosis)
        
        
if __name__ == '__main__':
    main()