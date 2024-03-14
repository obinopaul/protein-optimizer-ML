import pandas as pd          # data analysis library for handling structured data             
import numpy as np           # mathematical library for working with numerical data
import os 
import joblib               # library for saving and loading models in a file
import gradio as gr          # library for creating UI for your ML model 
from utils import *         # import the function from utils.py file

os.chdir(r"C:\Users\pault\OneDrive - University of Oklahoma\GRA - Bio-Manufacturing\1. ML-Cytovance-OU-Research") # Set the working directory to the folder where the model is saved

scale_cols = ['output_WCW_gl', 'output_agitation', 'output_air_%', 'output_D0_%', 'output_gasflow', 'output_O2', 'output_Ph', 
              'output_feed_%', 'output_feed', 'output_Temp', 'output_glycerol_gl', 
              'output_glucose_gl', 'output_acetate_mmol_l', 'output_phosphate_mmol_l']


def predict(time_point, output_WCW_gl, output_agitation, output_air_percent, output_D0_percent, output_gasflow, output_O2, output_Ph, output_feed_percent, output_feed, output_Temp, output_glycerol_gl, output_glucose_gl, output_acetate_mmol_l, output_phosphate_mmol_l):
    # Convert the time_point to a datetime index (adjust format as necessary)
    
   
    important_features = ['output_WCW_gl', 'output_phosphate_mmol_l', 'output_gasflow', 
                      'output_agitation', 'output_feed_%', 'output_glycerol_gl', 'output_glucose_gl', 
                      'input_Timepoint (hr)_sin', 'input_Timepoint (hr)_cos']
    
    # Create a DataFrame from the inputs, with time_point as the index
    df = pd.DataFrame({
        'input_Timepoint (hr)': [time_point],  # 'time_point' is the name of the time column in the model
        'output_WCW_gl': [output_WCW_gl],
        'output_agitation': [output_agitation],
        'output_air_%': [output_air_percent],
        'output_D0_%': [output_D0_percent],
        'output_gasflow': [output_gasflow],
        'output_O2': [output_O2],
        'output_Ph': [output_Ph],
        'output_feed_%': [output_feed_percent],
        'output_feed': [output_feed],
        'output_Temp': [output_Temp],
        'output_glycerol_gl': [output_glycerol_gl],
        'output_glucose_gl': [output_glucose_gl],
        'output_acetate_mmol_l': [output_acetate_mmol_l],
        'output_phosphate_mmol_l': [output_phosphate_mmol_l]
    })
    
    # Scale, Concat and add Cyclical Time Features
    scaler = joblib.load(r"models/preprocessing/Data 3/StandardScaler/StandardScaler_0.joblib")  # Function to load your saved model from a file
    df = add_cyclical_time_features(df, 'input_Timepoint (hr)', 48.0)   # Add cyclical features to the dataframe
    df = scale_and_concat(df, train = False, scaler = scaler)  # Apply the loaded model to new data.

    # Polynomial Feature Extraction
    poly = joblib.load(r"models/preprocessing/Data 3/PolynomialFeatures/PolynomialFeatures_0.joblib")      # Function to load your saved model from a file
    valid_columns = [col for col in important_features if col in df.columns] # Filter out columns in 'important_features' that are not present in 'df'
    poly_feature_names = ['poly_' + name for name in poly.get_feature_names_out(valid_columns)]
    poly_df = pd.DataFrame(poly.transform(df[valid_columns]), columns=poly_feature_names, index=df.index)  # Apply the loaded model to new data.
    df_final = pd.concat([df.drop(columns=valid_columns), poly_df], axis=1)  # Concatenate the polynomial features with the original DataFrame
    
    # Feature selection with RFE
    rfe = joblib.load(r"models/preprocessing/Data 3/RFE/RFE_0.joblib")  # Function to load your saved model from a file
    df_rfe = rfe.transform(df_final)   # Transform the dataframe using your trained model
    selected_features_1 = df_final.columns[rfe.support_]   # This will return an array of boolean values, where True indicates that the feature was selected
    df_final = pd.DataFrame(df_rfe, columns=selected_features_1, index=df_final.index)  # Convert the arrays back to DataFrames for compatibility and readability
        
    # Feature selection with model
    rfm = joblib.load(r"models/preprocessing/Data 3/SelectFromModel/SelectFromModel_0.joblib")  # Function to load your saved model from a file
    df_fs = rfm.transform(df_final)
    selected_features = df_final.columns[rfm.get_support()]
    df_fs = pd.DataFrame(df_fs, columns=selected_features, index=df_final.index)
    
        
    # Load your model and make predictions on the DataFrame
    model = joblib.load(r"models/Data 3/RandomForest_0.joblib")  # Function to load your saved model from a file
    y_pred = model.predict(df_fs)
    y_pred = y_pred.tolist()
    # For demonstration, let's just return the DataFrame as a string
    return f"{y_pred[0]:.2f}"



# Define the Gradio input components for each column
inputs = [
    # gr.Textbox(label="Time Point", placeholder="Enter time point in hrs (e.g., 2)"),
    gr.Slider(minimum=1, maximum=100, label = 'Time Point (hrs)'),
    gr.Number(label="WCW (g/L)"),
    gr.Number(label="Agitation"),
    gr.Number(label="Air %"),
    gr.Number(label="Disolved Oxygen Percentage (DO)"),
    gr.Number(label="Gas Flow"),
    gr.Number(label="Oxygen Percentage (O2)"),
    gr.Number(label="Ph"),
    gr.Number(label="Feed Percentage"),
    gr.Number(label="Feed"),
    # gr.Number(label="Temp"), 
    gr.Slider(minimum=0, maximum=100, label = 'Temperature (Celsius)'), 
    gr.Number(label="Glycerol (g/L)"),
    gr.Number(label="Glucose (g/L)"),
    gr.Number(label="Acetate (mmol/L)"),
    gr.Number(label="Phosphate (mmol/L)")
]

examples = [
    [14, 125, 1184.56,	94.78,	39.11,	4.92,	5.22,	6.74,	20.15,	0.6851,	29.95,	0.01,	1.01008,	1.33,	7.78],
    [28, 213.5,	1185.07, 89.96,	40.31,	5,	    10.04,	6.7,	17.9,	0.6086,	29.98,	0.01,	1.16514,	3.22,	0]
]
     
     
     
# Define the Gradio output component
output = gr.Textbox(label="OD600nm Prediction")
style = {'description': 'width: 50%'}



# Create the Gradio interface
iface = gr.Interface(fn=predict, inputs=inputs, outputs=output, examples = examples, 
                     title="Protein Titer OD600nm Prediction - Proof of Concept App", 
                     description="Enter the values for the specified Parameters along with the time point.",
                     theme="freddyaboulton/dracula_revamped")

# Launch the Gradio app
iface.launch()