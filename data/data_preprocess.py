"""Hide-and-Seek Privacy Challenge Codebase.

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar, 
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data," 
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Link: https://www.vanderschaar-lab.com/announcing-the-neurips-2020-hide-and-seek-privacy-challenge/

Last updated Date: Oct 17th 2020
Code author: Jinsung Yoon, Evgeny Saveliev
Contact: jsyoon0823@gmail.com, e.s.saveliev@gmail.com


-----------------------------

(1) data_preprocess: Load the data and preprocess into a 3d numpy array
(2) imputater: Impute missing data 
"""
# Local packages
import os
from typing import Union, Tuple, List
import warnings
warnings.filterwarnings("ignore")

# 3rd party modules
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def data_preprocess(
    file_name: str, 
    max_seq_len: int, 
    num_patients : int=17,  
    padding_value: float=-1.0,
    impute_method: str="mode", 
    scaling_method: str="minmax",  
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Load the data and preprocess into 3d numpy array.
    Preprocessing includes:
    1. Remove outliers
    2. Extract sequence length for each patient id
    3. Impute missing data 
    4. Normalize data
    6. Sort dataset according to sequence length

    Args:
    - file_name (str): CSV file name
    - max_seq_len (int): maximum sequence length
    - impute_method (str): The imputation method ("median" or "mode") 
    - scaling_method (str): The scaler method ("standard" or "minmax")

    Returns:
    - processed_data: preprocessed data
    - time: ndarray of ints indicating the length for each data
    - params: the parameters to rescale the data 
    """

    #########################
    # Load data
    #########################

    index = 'Idx'

    # Load csv
    print("Loading data...\n")
    if file_name=="stock":
        ori_data = pd.read_csv('./data/stock.csv')
        
    else: 
        ori_data = bg_level('./data', max_seq_len, num_patients)
        #dim = 1
    # Remove spurious column, so that column 0 is now 'admissionid'.
    if ori_data.columns[0] == "Unnamed: 0":  
        ori_data = ori_data.drop(["Unnamed: 0"], axis=1)
        ori_data.to_csv("./dropped.csv")
        
    #########################
    # Remove outliers from dataset
    #########################
    no = ori_data.shape[0]
    z_scores = stats.zscore(ori_data, axis=0, nan_policy='omit')
    z_filter = np.nanmax(np.abs(z_scores), axis=1) < 3
    ori_data = ori_data[z_filter]
    print(f"Dropped {no - ori_data.shape[0]} rows (outliers)\n")

    # Parameters
    if file_name=="stock":
        uniq_id = np.unique(ori_data[index])
        no = len(uniq_id)
        dim = len(ori_data.columns) - 1
    elif file_name=="bg_level":
        #uniq_id = np.unique(ori_data[index])
        no = ori_data.shape[0]
        dim = 1

    #########################
    # Impute, scale and pad data
    #########################
    
    # Initialize scaler
    if scaling_method == "minmax":
        scaler = MinMaxScaler()
        scaler.fit(ori_data)
        params = [scaler.data_min_, scaler.data_max_]
    
    elif scaling_method == "standard":
        scaler = StandardScaler()
        scaler.fit(ori_data)
        params = [scaler.mean_, scaler.var_]

    # Imputation values
    if impute_method == "median":
        impute_vals = ori_data.median()
    elif impute_method == "mode":
        impute_vals = stats.mode(ori_data).mode[0]
    else:
        raise ValueError("Imputation method should be `median` or `mode`")    

    # TODO: Sanity check for padding value
    # if np.any(ori_data == padding_value):
    #     print(f"Padding value `{padding_value}` found in data")
    #     padding_value = np.nanmin(ori_data.to_numpy()) - 1
    #     print(f"Changed padding value to: {padding_value}\n")
    print(ori_data.shape)
    # Output initialization
    output = np.empty([no, max_seq_len, dim])  # Shape:[no, max_seq_len, dim]
    output.fill(padding_value)
    print("output", output.shape)
    time = []

    if file_name == "bg_level":
        scaled_data = scaler.transform(ori_data)
        # For each uniq id
        for i in tqdm(range(no)):
            # Extract the time-series data with a certain admissionid
            curr_data = scaled_data[i, :]
            #print(curr_data.shape)
            curr_data = curr_data.reshape(-1, 1)
            #print(curr_data.shape)
            # Impute missing data
            curr_data = imputer(curr_data, impute_vals)
            #print(curr_data.shape)
            # Normalize data
            #curr_data = scaler.transform(curr_data)
            
            # Extract time and assign to the preprocessed data (Excluding ID)
            curr_no = len(curr_data)

            # Pad data to `max_seq_len`
            if curr_no >= max_seq_len:
                output[i, :, :] = curr_data[:max_seq_len, :]  # Shape: [1, max_seq_len, dim]
                time.append(max_seq_len)
            else:
                output[i, :curr_no, :] = curr_data[:, :]  # Shape: [1, max_seq_len, dim]
                time.append(curr_no)
        
        # Output initialization
        ori_output = np.empty([no, max_seq_len, dim])  # Shape:[no, max_seq_len, dim]
        ori_output.fill(padding_value)
        #time = []
        # For each uniq id
        for i in tqdm(range(no)):
            # Extract the time-series data with a certain admissionid

            curr_data = ori_data.to_numpy()[i, :]

            # Impute missing data
            curr_data = imputer(curr_data, impute_vals)

            # Normalize data
            #curr_data = scaler.transform(curr_data)
            
            # Extract time and assign to the preprocessed data (Excluding ID)
            curr_no = len(curr_data)

            # Pad data to `max_seq_len`
            if curr_no >= max_seq_len:
                ori_output[i, :, :] = curr_data[:max_seq_len, :]  # Shape: [1, max_seq_len, dim]
                #time.append(max_seq_len)
            else:
                ori_output[i, :curr_no, :] = curr_data[:, :]  # Shape: [1, max_seq_len, dim]
                #time.append(curr_no)
        print(output.shape)
    
    if file_name == "stock":
        #scaled_data = scaler.transform(ori_data)
        # For each uniq id
        for i in tqdm(range(no)):
            # Extract the time-series data with a certain admissionid
            curr_data = ori_data[ori_data[index]==uniq_id[i]].to_numpy()
            #print(curr_data.shape)
            #curr_data = curr_data.reshape(-1, 1)
            #print(curr_data.shape)
            # Impute missing data
            curr_data = imputer(curr_data, impute_vals)
            #print(curr_data.shape)
            # Normalize data
            curr_data = scaler.transform(curr_data)
            
            # Extract time and assign to the preprocessed data (Excluding ID)
            curr_no = len(curr_data)

            # Pad data to `max_seq_len`
            if curr_no >= max_seq_len:
                output[i, :, :] = curr_data[:max_seq_len, 1:]  # Shape: [1, max_seq_len, dim]
                time.append(max_seq_len)
            else:
                output[i, :curr_no, :] = curr_data[:, 1:]  # Shape: [1, max_seq_len, dim]
                time.append(curr_no)
        
        # Output initialization
        ori_output = np.empty([no, max_seq_len, dim])  # Shape:[no, max_seq_len, dim]
        ori_output.fill(padding_value)
        time = []
        # For each uniq id
        for i in tqdm(range(no)):
            # Extract the time-series data with a certain admissionid

            curr_data = ori_data[ori_data[index]==uniq_id[i]].to_numpy()

            # Impute missing data
            curr_data = imputer(curr_data, impute_vals)

            # Normalize data
            #curr_data = scaler.transform(curr_data)
            
            # Extract time and assign to the preprocessed data (Excluding ID)
            curr_no = len(curr_data)

            # Pad data to `max_seq_len`
            if curr_no >= max_seq_len:
                ori_output[i, :, :] = curr_data[:max_seq_len, 1:]  # Shape: [1, max_seq_len, dim]
                #time.append(max_seq_len)
            else:
                ori_output[i, :curr_no, :] = curr_data[:, 1:]  # Shape: [1, max_seq_len, dim]
                #time.append(curr_no)
        print(output.shape)

    return ori_output, output, time, params, max_seq_len, padding_value, scaler

  
def bg_level(data_path, seq_len, num_patients):
    data = []
    for i in range(num_patients):
        p_data = np.loadtxt(data_path + '/BG_data/V_CGMS_{}.csv'.format(i+1), delimiter=",")
        #print(p_data)
        #temp_data = []
        # cut data by sequence length
        for j in range(0, len(p_data) - seq_len):
            #x = [i]
            #print(x)
            #print(p_data[j:j + seq_len].tolist())
            #x = np.array(x + p_data[j:j + seq_len].tolist())
            #print(x)
            data.append(p_data[j:j + seq_len])
    #print(train_data)
    # Mix the datasets (to make it similar to i.i.d)
    #idx = np.random.permutation(len(train_data))    
    #data = []
    #for i in range(len(train_data)):
    #    data.append(train_data[idx[i]])
    data = np.array(data)
    #data = data.reshape((data.shape[0], data.shape[1], 1))
    data = pd.DataFrame(data)
    #data.rename(columns={0:'Idx'}, inplace=True)
    print(data.shape)
    #data.to_csv("./data/bg.csv")
    return data
    
    

def imputer(
    curr_data: np.ndarray, 
    impute_vals: List, 
    zero_fill: bool = True
) -> np.ndarray:
    """Impute missing data given values for each columns.

    Args:
        curr_data (np.ndarray): Data before imputation.
        impute_vals (list): Values to be filled for each column.
        zero_fill (bool, optional): Whather to Fill with zeros the cases where 
            impute_val is nan. Defaults to True.

    Returns:
        np.ndarray: Imputed data.
    """

    curr_data = pd.DataFrame(data=curr_data)
    impute_vals = pd.Series(impute_vals)
    
    # Impute data
    imputed_data = curr_data.fillna(impute_vals)

    # Zero-fill, in case the `impute_vals` for a particular feature is `nan`.
    imputed_data = imputed_data.fillna(0.0)

    # Check for any N/A values
    if imputed_data.isnull().any().any():
        raise ValueError("NaN values remain after imputation")

    return imputed_data.to_numpy()
