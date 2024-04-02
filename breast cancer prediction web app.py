
import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/pc/Desktop/projects - manasvi/Breast Cancer Classification using ML/trained_model.sav', 'rb'))

# creating a function for prediction

def breast_cancer_classification(input_data):
    
    # change the input data to a numpy array
    input_np_array = np.asarray(input_data)

    # reshape the numpy array as we are predicting for one datapoint
    input_data_reshaped = input_np_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction[0])

    if (prediction[0] == 0):
      return 'Patient has Breast Cancer'

    else:
      return 'Patient does not have Breast Cancer'
  

def main():
    
    # giving a title
    st.title('Breast Cancer Prediction Web App')
    
    # getting the input data from the user
    # "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"  
    radius_mean = st.text_input('Radius Mean ')
    texture_mean = st.text_input('Texture Mean ')
    perimeter_mean = st.text_input('Perimeter Mean ')
    area_mean = st.text_input('Area Mean ')
    smoothness_mean = st.text_input('Smoothness Mean ')
    compactness_mean = st.text_input('Compactness Mean ')
    concavity_mean = st.text_input('Concavity Mean ')
    concave_points_mean = st.text_input('Concave Points Mean ')
    symmetry_mean = st.text_input('Symmetry Mean ')
    fractal_dimension_mean = st.text_input('Fractal Dimension Mean ')
    radius_se = st.text_input('Radius Standard Error ')
    texture_se = st.text_input('Texture Standard Error ')
    perimeter_se = st.text_input('Perimeter Standard Error ')
    area_se = st.text_input('Area Standard Error ')
    smoothness_se = st.text_input('Smoothness Standard Error ')
    compactness_se = st.text_input('Compactness Standard Error ')
    concavity_se = st.text_input('Concavity Standard Error ')
    concave_points_se = st.text_input('Concave Points Standard Error ')
    symmetry_se = st.text_input('Symmetry Standard Error ')
    fractal_dimension_se = st.text_input('Fractal Dimension Standard Error ')
    radius_worst = st.text_input('Radius Largest Value ')
    texture_worst = st.text_input('Texture Largest Value ')
    perimeter_worst = st.text_input('Perimeter Largest Value ')
    area_worst = st.text_input('Area Largest Value ')
    smoothness_worst = st.text_input('Smoothness Largest Value ')
    compactness_worst = st.text_input('Compactness Largest Value ')
    concavity_worst = st.text_input('Concavity Largest Value ')
    concave_points_worst = st.text_input('Concave Points Largest Value ')
    symmetry_worst = st.text_input('Symmetry Largest Value ')
    fractal_dimension_worst = st.text_input('Fractal Dimension Largest Value ')
    
    # code for prediction
    diagnosis = ''
    
    # creating a button for predicting
    
    if st.button("Test Result"):
        diagnosis = breast_cancer_classification([float(radius_mean), float(texture_mean), float(perimeter_mean), float(area_mean), float(smoothness_mean), float(compactness_mean), float(concavity_mean), float(concave_points_mean), float(symmetry_mean), float(fractal_dimension_mean), float(radius_se), float(texture_se), float(perimeter_se), float(area_se), float(smoothness_se), float(compactness_se), float(concavity_se), float(concave_points_se), float(symmetry_se), float(fractal_dimension_se), float(radius_worst), float(texture_worst), float(perimeter_worst), float(area_worst), float(smoothness_worst), float(compactness_worst), float(concavity_worst), float(concave_points_worst), float(symmetry_worst), float(fractal_dimension_worst)])
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
