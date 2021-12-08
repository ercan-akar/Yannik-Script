#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from os import listdir
import os
from scipy import signal
import pickle
import seaborn as sns
# import xlsxwriter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error

def prepare_dataset(df, batch_ID_input):
    # db = pd.read_excel(int_file_dataset, sheet_name = int_sheet_dataset)
    db = df
    db.columns = db.iloc[1]
    db.drop(index=[0,1,2], inplace=True)
    db_predict = db[db['Lbl_6'].isin(batch_ID_input)]
    db_predict = db_predict.reset_index(drop=True)
    return db_predict

def predict(df, Intel_num, project_path, remove_outlier):
    # db_predict = intel_predict_dataset_loading(int_file_dataset, int_sheet_dataset, batch_ID_input)
    db_predict = df
    
    prediction_result_dir = os.path.join(project_path, 'predictions')
    data_dir = os.path.join(project_path, 'predictions', 'data')
    for directory in [prediction_result_dir, data_dir]:
        print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

    model_scaler_dir = os.path.join(project_path, 'modelling', 'scalers')
    model_pls_dir = os.path.join(project_path, 'modelling', 'models', 'PLS')
    model_rf_dir = os.path.join(project_path, 'modelling', 'models', 'RF')
    model_xgb_dir = os.path.join(project_path, 'modelling', 'models', 'XGB')
    model_graphics_pls_dir = os.path.join(project_path, 'modelling', 'graphics', 'PLS')
    model_graphics_rf_dir = os.path.join(project_path, 'modelling', 'graphics', 'RF')
    model_graphics_xgb_dir = os.path.join(project_path, 'modelling', 'graphics', 'XGB')
    model_excel_pls_dir = os.path.join(project_path, 'modelling', 'excel', 'PLS')
    model_excel_rf_dir = os.path.join(project_path, 'modelling', 'excel', 'RF')
    model_excel_xgb_dir = os.path.join(project_path, 'modelling', 'excel', 'XGB')
    model_data_dir = os.path.join(project_path, 'modelling', 'data')

    data_ALL_QC            = pd.read_excel(os.path.join(model_data_dir, 'data_ALL_QC.xlsx'))
    data_PP_RM             = pd.read_excel(os.path.join(model_data_dir, 'data_PP_RM.xlsx'))
    data_ALL_QC_until_bulk = pd.read_excel(os.path.join(model_data_dir, 'data_ALL_QC_until_bulk.xlsx'))
    data_PP_RM_until_bulk  = pd.read_excel(os.path.join(model_data_dir, 'data_PP_RM_until_bulk.xlsx'))
    
    pls_model_all_qc_filename            = os.path.join(model_pls_dir, 'ALL_QC.pkl')
    pls_model_pp_rm_filename             = os.path.join(model_pls_dir, 'PP_RM.pkl')
    pls_model_all_qc_until_bulk_filename = os.path.join(model_pls_dir, 'ALL_QC_until_bulk.pkl')
    pls_model_pp_rm_until_bulk_filename  = os.path.join(model_pls_dir, 'PP_RM_until_bulk.pkl')

    rf_model_all_qc_filename            = os.path.join(model_rf_dir, 'ALL_QC.pkl')
    rf_model_pp_rm_filename             = os.path.join(model_rf_dir, 'PP_RM.pkl')
    rf_model_all_qc_until_bulk_filename = os.path.join(model_rf_dir, 'ALL_QC_until_bulk.pkl')
    rf_model_pp_rm_until_bulk_filename  = os.path.join(model_rf_dir, 'PP_RM_until_bulk.pkl')

    xgb_model_all_qc_filename            = os.path.join(model_xgb_dir, 'ALL_QC.pkl')
    xgb_model_pp_rm_filename             = os.path.join(model_xgb_dir, 'PP_RM.pkl')
    xgb_model_all_qc_until_bulk_filename = os.path.join(model_xgb_dir, 'ALL_QC_until_bulk.pkl')
    xgb_model_pp_rm_until_bulk_filename  = os.path.join(model_xgb_dir, 'PP_RM_until_bulk.pkl')

    scaler_all_qc_filename            = os.path.join(model_scaler_dir, 's_ALL_QC.pkl')
    scaler_pp_rm_filename             = os.path.join(model_scaler_dir, 's_PP_RM.pkl')
    scaler_all_qc_until_bulk_filename = os.path.join(model_scaler_dir, 's_ALL_QC_until_bulk.pkl')
    scaler_pp_rm_until_bulk_filename  = os.path.join(model_scaler_dir, 's_PP_RM_until_bulk.pkl')
    scaler_y_filename                 = os.path.join(model_scaler_dir, 's_y.pkl')
    
    pls_model_ALL_QC = pickle.load(open(pls_model_all_qc_filename, 'rb'))
    pls_model_PP_RM = pickle.load(open(pls_model_pp_rm_filename, 'rb'))
    pls_model_ALL_QC_until_bulk = pickle.load(open(pls_model_all_qc_until_bulk_filename, 'rb'))
    pls_model_PP_RM_until_bulk = pickle.load(open(pls_model_pp_rm_until_bulk_filename, 'rb'))
    rf_model_ALL_QC = pickle.load(open(rf_model_all_qc_filename, 'rb'))
    rf_model_PP_RM = pickle.load(open(rf_model_pp_rm_filename, 'rb'))
    rf_model_ALL_QC_until_bulk = pickle.load(open(rf_model_all_qc_until_bulk_filename, 'rb'))
    rf_model_PP_RM_until_bulk = pickle.load(open(rf_model_pp_rm_until_bulk_filename, 'rb'))
    xgb_model_ALL_QC = pickle.load(open(xgb_model_all_qc_filename, 'rb'))
    xgb_model_PP_RM = pickle.load(open(xgb_model_pp_rm_filename, 'rb'))
    xgb_model_ALL_QC_until_bulk = pickle.load(open(xgb_model_all_qc_until_bulk_filename, 'rb'))
    xgb_model_PP_RM_until_bulk = pickle.load(open(xgb_model_pp_rm_until_bulk_filename, 'rb'))
    s_ALL_QC = pickle.load(open(scaler_all_qc_filename, 'rb'))
    s_PP_RM = pickle.load(open(scaler_pp_rm_filename, 'rb'))
    s_ALL_QC_until_bulk = pickle.load(open(scaler_all_qc_until_bulk_filename, 'rb'))
    s_PP_RM_until_bulk = pickle.load(open(scaler_pp_rm_until_bulk_filename, 'rb'))
    s_y = pickle.load(open(scaler_y_filename, 'rb'))   
    
    db_ALL_QC = db_predict[data_ALL_QC.columns].copy()
    db_ALL_QC_until_bulk = db_predict[data_ALL_QC_until_bulk.columns].copy()
    db_PP_RM = db_predict[data_PP_RM.columns].copy()
    db_PP_RM_until_bulk = db_predict[data_PP_RM_until_bulk.columns].copy()
    
    db_models = [db_ALL_QC, db_PP_RM, db_ALL_QC_until_bulk, db_PP_RM_until_bulk]
    data_models = [data_ALL_QC, data_PP_RM, data_ALL_QC_until_bulk, data_PP_RM_until_bulk]
    
    #filling empty columns
    for i in range(len(db_models)):
        for column in db_models[i].columns:
            db_models[i][column].fillna(data_models[i][column].mean(), inplace=True)
            
    # Min Max Scaling
    X_ALL_QC = db_ALL_QC.copy()
    X_PP_RM = db_PP_RM.copy()
    X_ALL_QC_until_bulk = db_ALL_QC_until_bulk.copy()
    X_PP_RM_until_bulk = db_PP_RM_until_bulk.copy()
    scaler_X = [s_ALL_QC, s_PP_RM, s_ALL_QC_until_bulk, s_PP_RM_until_bulk]
    X_set = [X_ALL_QC, X_PP_RM, X_ALL_QC_until_bulk, X_PP_RM_until_bulk]

    for i in range(len(scaler_X)):
        X_scaled = scaler_X[i].transform(X_set[i])
        X_set[i] = pd.DataFrame(X_scaled, columns = X_set[i].columns)
        
    #outlier check
    outliers_batch_number = []
    outliers_list = []
    
    for i in range(len(X_set)):
        for column in X_set[i].columns:
            for j in range(X_set[i].shape[0]):
                if (X_set[i].loc[j,column] < 0) or (X_set[i].loc[j,column] > 1):
                    outliers_batch_number.append(j)
                    outliers_list.append(str(column)+'_'+str(db_predict.loc[j,'Lbl_6']))
                    
    if remove_outlier == 0: ##remove outlier batches
        unique_batch_number = sorted(list(set(outliers_batch_number)))
        for i in range(len(X_set)):
            X_set[i].drop(index=unique_batch_number, inplace=True)
            
    if remove_outlier == 1: ##consider outlier batches
        pass
    
    complete_models = [pls_model_ALL_QC, pls_model_PP_RM , pls_model_ALL_QC_until_bulk, pls_model_PP_RM_until_bulk, rf_model_ALL_QC , rf_model_PP_RM, rf_model_ALL_QC_until_bulk, rf_model_PP_RM_until_bulk, xgb_model_ALL_QC, xgb_model_PP_RM , xgb_model_ALL_QC_until_bulk, xgb_model_PP_RM_until_bulk]
    names_models = ['pls_model_ALL_QC', 'pls_model_PP_RM', 'pls_model_ALL_QC_until_bulk', 'pls_model_PP_RM_until_bulk', 'rf_model_ALL_QC', 'rf_model_PP_RM', 'rf_model_ALL_QC_until_bulk', 'rf_model_PP_RM_until_bulk', 'xgb_model_ALL_QC', 'xgb_model_PP_RM', 'xgb_model_ALL_QC_until_bulk', 'xgb_model_PP_RM_until_bulk']    
    
    y_models = {}
    
    for i in range(len(X_set)):
        y_pls = complete_models[i].predict(X_set[i]).reshape(-1, 1)
        y_rf = complete_models[i+4].predict(X_set[i]).reshape(-1, 1)
        y_xgb = complete_models[i+8].predict(X_set[i]).reshape(-1, 1)
        
        y_pls = s_y.inverse_transform(y_pls)
        y_rf = s_y.inverse_transform(y_rf)
        y_xgb = s_y.inverse_transform(y_xgb)
        
        y_models['y'+names_models[i]] = y_pls
        y_models['y'+names_models[i+4]] = y_rf
        y_models['y'+names_models[i+8]] = y_xgb
        
    df_y_pred = pd.DataFrame([y_models])
    
    return df_y_pred

    df_y_pred.to_excel(os.path.join(prediction_result_dir, 'predictions_results.xlsx'), index=False)

if __name__ == '__main__':
    int_file_dataset = './Intel_Master_dataset/PHPS_DATA_SET_P2_Intel20211027.xlsx'
    int_sheet_dataset = 'Dataset'
    Intel_num = 'Intel_2'
    path_directory = os.path.join('C:\\', 'Users', 'M304498', 'test_project')
    batch_ID_input = ['21090196D','21090796D','21090802D','21100702D','21082699U','21091599D','21100699D','21081995U','21100695U']
    remove_outlier = 0

    df = pd.read_excel(int_file_dataset, sheet_name = None)[int_sheet_dataset]
    df_prepared = prepare_dataset(df, batch_ID_input)

    predicted = predict(Intel_num, path_directory, remove_outlier)
