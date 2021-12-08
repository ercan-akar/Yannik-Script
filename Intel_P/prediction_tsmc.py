#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from os import listdir
from scipy import signal
import pickle
import seaborn as sns
import xlsxwriter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error


tsmc_file_dataset = './TSMC_Master_dataset/Data_SA_02.xlsx'
tsmc_sheet_dataset = 'FG_ECD'
label = 'co_sel_ecd'
path_directory = 'TSMC_Results_dataset'
batch_ID_input = ['SHBN6809 (G5C1)','SHBN8615 (G5C2)','SHBN8227 (J5D1)']

def prepare_data(df, batch_ID_input):
    # db = pd.read_excel(tsmc_file_dataset, sheet_name = tsmc_sheet_dataset)
    db = df
    db_predict = db[db['Parameter'].isin(batch_ID_input)]
    db_predict = db_predict.reset_index(drop=True)
    return db_predict

def predict(df, label, path_directory):
    # db_predict = tsmc_predict_dataset_loading(tsmc_file_dataset, tsmc_sheet_dataset, batch_ID_input)
    db_predict = df
    
    data_x = pd.read_excel('./'+str(path_directory)+'data_X.xlsx')
    pls_model_filename = './'+str(path_directory)+'/pls_model_'+str(label)'.pkl'
    rf_model_filename = './'+str(path_directory)+'/rf_model_'+str(label)+'.pkl'
    xgb_model_filename = './'+str(path_directory)+'/xgb_model_'+str(label)+'.pkl'
    scaler_x_filename = './'+str(path_directory)+'/s_x'+str(label)+'.pkl'
    scaler_y_filename = './'+str(path_directory)+'/s_y'+str(label)+'.pkl'  
    
    pls_model = pickle.load(open(pls_model_filename, 'rb'))
    rf_model = pickle.load(open(rf_model_filename, 'rb'))
    xgb_model = pickle.load(open(xgb_model_filename, 'rb'))
    s_x = pickle.load(open(scaler_x_filename, 'rb'))
    s_y = pickle.load(open(scaler_y_filename, 'rb'))
    
    db_x = db_predict[data_x.columns].copy()
    
    #filling empty columns
    for column in db_x.columns:
        db_x[column].fillna(data_x[column].mean(), inplace=True)
        
    # Min Max Scaling
    X = db_x.copy()
    
    X_scaled = s_x.transform(X)
    X = pd.DataFrame(X_scaled, columns = X.columns)
    
    #outlier check
    #outliers_batch_number = []
    outliers_list = []
    
    for column in X.columns:
        for j in range(X.shape[0]):
            if (X.loc[j,column] < 0) or (X.loc[j,column] > 1):
                #outliers_batch_number.append(j)
                outliers_list.append(str(column)+'_'+str(db_predict.loc[j,'Parameter']))
     

    complete_models = [pls_model, rf_model, xgb_model]
    names_models = ['pls_model', 'rf_model', 'xgb_model']    
    
    y_models = {}
    
    for i in range(len(complete_models)):
        y_ = complete_models[i].predict(X).reshape(-1, 1)
        y_ = s_y.inverse_transform(y_)
        
        y_models['y'+names_models[i]] = y_
        
    df_y_pred = pd.DataFrame([y_models])
    
    #df_y_pred.to_excel('predictions_results_'+str(label)+'.xlsx', index=False)

