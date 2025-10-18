# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle 
loaded_model=pickle.load(open('C:/Users/USER/Desktop/Deploying Machine learning Model/trained_model.sav','rb'))


input_data=(4,110,92,0,0,37.6,0.191,30)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
    print("la personne n'est pas diabétique")
else:
    print("la personne est diabétique")
    