# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('C:/Users/pc/Desktop/projects - manasvi/Breast Cancer Classification using ML/trained_model.sav', 'rb'))

input_data = (12.45,15.7,82.57,477.1,0.1278,0.17,0.1578,0.08089,0.2087,0.07613,0.3345,0.8902,2.217,27.19,0.00751,0.03345,0.03672,0.01137,0.02165,0.005082,15.47,23.75,103.4,741.6,0.1791,0.5249,0.5355,0.1741,0.3985,0.1244)

# change the input data to a numpy array
input_np_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_np_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction[0])

if (prediction[0] == 0):
  print('Patient has Breast Cancer')

else:
  print('Patient does not have Breast Cancer')
