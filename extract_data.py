import pandas as pd
import numpy as np
import os 

# Set working directory
os.chdir(r"C:\Users\pault\OneDrive - University of Oklahoma\GRA - Bio-Manufacturing\1. ML-Cytovance-OU-Research")


#########################################################################################################
# PROJECTG_Data EXCEL DATASET
#########################################################################################################

file_url = r"C:\Users\pault\OneDrive - University of Oklahoma\GRA - Bio-Manufacturing\1. ML-Cytovance-OU-Research\data\raw\ProjectG_Data.xlsx"
output_path = r"C:\Users\pault\OneDrive - University of Oklahoma\GRA - Bio-Manufacturing\1. ML-Cytovance-OU-Research\data\processed"


def load_data(file_url, sheet_name = None):
    df = pd.read_excel(file_url, sheet_name=sheet_name, header=2)
    df.drop(["Unnamed: 15", "Timepoint (hr).1", "Production day", "Unnamed: 33"], axis=1, inplace=True) # remove an empty column
    columns = df.columns.to_list()
    input_columns = ['input_' + col for col in columns[:14]]  # Rename columns up to 'OD600'
    output_columns = ['output_' + col for col in columns[14:]]  # Rename columns from 'OD600' to the end
    new_columns = input_columns + output_columns
    df.columns = new_columns
    return df

def save_csv (df, filename:str, index = False):
    df.to_csv(output_path + '\\' + filename + '.csv', index = index)



# Load and save data in sheet 1 - ProjectG_Data
df_210623 = load_data(file_url, sheet_name=0)

exp_210623_1 = df_210623.iloc[:22,]
exp_210623_2 = df_210623.iloc[22:44,]
exp_210623_3 = df_210623.iloc[44:66,]
exp_210623_4 = df_210623.iloc[66:88,]

save_csv(exp_210623_1, 'exp_210623_1')
save_csv(exp_210623_2, 'exp_210623_2')
save_csv(exp_210623_3, 'exp_210623_3')
save_csv(exp_210623_4, 'exp_210623_4')


# Load and save data in sheet 2 - ProjectG_Data
def load_data_2(file_url, sheet_name = None):
    df = pd.read_excel(file_url, sheet_name=sheet_name, header=2)
    df.drop(["Unnamed: 15", "Timepoint (hr).1", "Production day"], axis=1, inplace=True) # remove an empty column
    columns = df.columns.to_list()
    input_columns = ['input_' + col for col in columns[:14]]  # Rename columns up to 'OD600'
    output_columns = ['output_' + col for col in columns[14:]]  # Rename columns from 'OD600' to the end
    new_columns = input_columns + output_columns
    df.columns = new_columns
    return df


# load data 
df_211130 = load_data_2(file_url, sheet_name=1)
exp_211130_1 = df_211130.iloc[:21,]
exp_211130_2 = df_211130.iloc[21:42,]
exp_211130_3 = df_211130.iloc[42:63,]
exp_211130_4 = df_211130.iloc[63:84,] 

# save data
save_csv(exp_211130_1, 'exp_211130_1')
save_csv(exp_211130_2, 'exp_211130_2')
save_csv(exp_211130_3, 'exp_211130_3')
save_csv(exp_211130_4, 'exp_211130_4')


# Load and save data in Sheet 3 - ProjectG_Data
df_211013 = load_data_2(file_url, sheet_name=2)
exp_211013_1 = df_211013.iloc[:19,]
exp_211013_2 = df_211013.iloc[19:38,]
exp_211013_3 = df_211013.iloc[38:57,]
exp_211013_4 = df_211013.iloc[57:76]

save_csv(exp_211013_1, 'exp_211013_1')
save_csv(exp_211013_2, 'exp_211013_2')
save_csv(exp_211013_3, 'exp_211013_3')
save_csv(exp_211013_4, 'exp_211013_4')


# Load and save data in Sheet 4 - ProjectG_Data
df_220822 = load_data_2(file_url, sheet_name=3)

exp_220822_1 = df_220822.iloc[:20,]
exp_220822_2 = df_220822.iloc[20:40,]
exp_220822_3 = df_220822.iloc[40:60,]
exp_220822_4 = df_220822.iloc[60:80,]

save_csv(exp_220822_1, 'exp_220822_1')
save_csv(exp_220822_2, 'exp_220822_2')
save_csv(exp_220822_3, 'exp_220822_3')
save_csv(exp_220822_4, 'exp_220822_4')




#########################################################################################################
# PROJECT_S EXCEL DATASET
#########################################################################################################
    
    
# load and save data in sheet 1 - Project_S
file_url_2 = r"C:\Users\pault\OneDrive - University of Oklahoma\GRA - Bio-Manufacturing\1. ML-Cytovance-OU-Research\data\raw\Project_S.xlsx"


def load_data_3(file_url, sheet_name = None, last_col_drop = None):
    df = pd.read_excel(file_url, sheet_name=sheet_name, header=2)
    df.drop(["Unnamed: 15", "Timepoint (hr).1", last_col_drop, "Production day",], axis=1, inplace=True) # remove an empty column
    df['Temp'] = df['Temp'].apply(lambda x: x.replace('±1oC', '')) # remove the °C symbol from temperature values
    df['pH setpoint'] = df['pH setpoint'].apply(lambda x: x.replace('±0.1', '')) # remove the ±0.1 from pH setpoint values
    columns = df.columns.to_list()
    input_columns = ['input_' + col for col in columns[:14]]  # Rename columns up to 'OD600'
    output_columns = ['output_' + col for col in columns[14:]]  # Rename columns from 'OD600' to the end
    new_columns = input_columns + output_columns
    df.columns = new_columns
    return df

# Load data and split into train/test sets (70%/30%)
df_220315c1 = load_data_3(file_url_2, sheet_name=0, last_col_drop = 'Unnamed: 33')

exp_220315c1_1 = df_220315c1.iloc[:19,]
exp_220315c1_2 = df_220315c1.iloc[19:38,]
exp_220315c1_3 = df_220315c1.iloc[38:57,]
exp_220315c1_4 = df_220315c1.iloc[57:76,]
exp_220315c1_5 = df_220315c1.iloc[76:95,]
exp_220315c1_6 = df_220315c1.iloc[95:114,]

  
save_csv(exp_220315c1_1, 'exp_220315c1_1')
save_csv(exp_220315c1_2, 'exp_220315c1_2')
save_csv(exp_220315c1_3, 'exp_220315c1_3')
save_csv(exp_220315c1_4, 'exp_220315c1_4')
save_csv(exp_220315c1_5, 'exp_220315c1_5')
save_csv(exp_220315c1_6, 'exp_220315c1_6')



# Load and save data in sheet 2 - Project_S
df_220329c2 = load_data_3(file_url_2, sheet_name=1, last_col_drop = "Unnamed: 35")

exp_220329c2_1 = df_220329c2.iloc[:25,]  # first row to 19th row
exp_220329c2_2 = df_220329c2.iloc[25:50,]  # 19th row to 38th row
exp_220329c2_3 = df_220329c2.iloc[50:75,]  # 38th row to 57th row
exp_220329c2_4 = df_220329c2.iloc[75:100,]  # 57th row to 76th row
exp_220329c2_5 = df_220329c2.iloc[100:125,]  # 76th row to 95th row
exp_220329c2_6 = df_220329c2.iloc[125:150,]  # 95th row to 114th row


save_csv(exp_220329c2_1, 'exp_220329c2_1')
save_csv(exp_220329c2_2, 'exp_220329c2_2')
save_csv(exp_220329c2_3, 'exp_220329c2_3')
save_csv(exp_220329c2_4, 'exp_220329c2_4')
save_csv(exp_220329c2_5, 'exp_220329c2_5')
save_csv(exp_220329c2_6, 'exp_220329c2_6')



# Load and save data in sheet 3 - Project_S
def load_data_3(file_url, sheet_name = None):
    df = pd.read_excel(file_url, sheet_name=sheet_name, header=2)
    df.drop(["Unnamed: 15", "Timepoint (hr).1", "Production day",], axis=1, inplace=True) # remove an empty column
    df['Temp (oC)'] = df['Temp (oC)'].apply(lambda x: x.replace('±1', '')) # remove the °C symbol from temperature values
    df['pH setpoint'] = df['pH setpoint'].apply(lambda x: x.replace('±0.1', '')) # remove the ±0.1 from pH setpoint values
    columns = df.columns.to_list()
    input_columns = ['input_' + col for col in columns[:14]]  # Rename columns up to 'OD600'
    output_columns = ['output_' + col for col in columns[14:]]  # Rename columns from 'OD600' to the end
    new_columns = input_columns + output_columns
    df.columns = new_columns
    return df


df_220309demo = load_data_3(file_url_2, sheet_name=2)

exp_220309demo_1 = df_220309demo.iloc[:23,]  # first row to 19th row
exp_220309demo_2 = df_220309demo.iloc[23:46,]  # 19th row to 38th row
exp_220309demo_3 = df_220309demo.iloc[46:69,]  # 38th row to 57th row
exp_220309demo_4 = df_220309demo.iloc[69:92,]  # 57th row to 76th row


save_csv(exp_220309demo_1, 'exp_220309demo_1')
save_csv(exp_220309demo_2, 'exp_220309demo_2')
save_csv(exp_220309demo_3, 'exp_220309demo_3')
save_csv(exp_220309demo_4, 'exp_220309demo_4') 

