#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from os import listdir, path
from scipy import signal
import pickle
import seaborn as sns
# import xlsxwriter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression


def prepare_main_dataset(df):
    df.columns = df.iloc[1]
    df.drop(index=[0,1,2], inplace=True)
    df = df.reset_index(drop=True)
    return df

def prepare_exclude_dataset(df):
    df_exclude = df[df.loc[:, 'Primary_ID'].notnull()]
    df_exclude = df_exclude.reset_index(drop=True)
    return df_exclude

def intel_main_dataset_loading(int_file_dataset, int_sheet_dataset):
    dz_1 = pd.read_excel(int_file_dataset, sheet_name = int_sheet_dataset)
    dz_1.columns = dz_1.iloc[1]
    dz_1.drop(index=[0,1,2], inplace=True)
    dz_1 = dz_1.reset_index(drop=True)
    return dz_1
    
def intel_exclude_dataset_loading(int_file_exclude, int_sheet_exclude):
    dz_2 = pd.read_excel(int_file_exclude, sheet_name = int_sheet_exclude)
    dz_2_exclude = dz_2[dz_2['Primary_ID'].notnull()]
    dz_2_exclude = dz_2_exclude.reset_index(drop=True)
    return dz_2_exclude

product_mapping = {
    'SP710': {
        'SP_list': ['SP710N', 'Spinfil71001N'],
        'Intel_num': 'Intel_2'
    },
    'SP730': {
        'SP_list': ['SP730N', 'Spinfil73001N'],
        'Intel_num': 'Intel_2'
    },
    'SP740': {
        'SP_list': ['Spinfil74001N'],
        'Intel_num': 'Intel_2'
    },
    'SP750': {
        'SP_list': ['SP750', 'SP750N', 'Spinfil75001N'],
        'Intel_num': 'Intel_3'
    },
    'SP790': {
        'SP_list': ['SP790N', 'Spinfil79001N'],
        'Intel_num': 'Intel_6'
    },
}

def clean_dataset(Product_name, df_main, df_exclude): #  int_file_dataset, int_sheet_dataset, int_file_exclude, int_sheet_exclude):
    db = df_main
    dc_exclude = df_exclude
    #db = intel_main_dataset_loading(int_file_dataset, int_sheet_dataset)
    #dc_exclude = intel_exclude_dataset_loading(int_file_exclude, int_sheet_exclude)
    ##identiy product name
    #if Product_name == 'SP710':
        #SP_list = ['SP710N', 'Spinfil71001N']
        #Intel_num = 'Intel_2'
    
    #if Product_name == 'SP730':
        #SP_list = ['SP730N', 'Spinfil73001N']
        #Intel_num = 'Intel_2'
    
    #if Product_name == 'SP740':
        #SP_list = ['Spinfil74001N']
        #Intel_num = 'Intel_2'
    
    #if Product_name == 'SP750':
        #SP_list = ['SP750', 'SP750N', 'Spinfil75001N']
        #Intel_num = 'Intel_3'
    
    #if Product_name == 'SP790':
        #SP_list = ['SP790N', 'Spinfil79001N']
        #Intel_num = 'Intel_6'
    SP_list = product_mapping[Product_name]['SP_list']
    Intel_num = product_mapping[Product_name]['Intel_num']
    
    db_700s = db[db['Lbl_2'].isin(SP_list)]
    db_700s = db_700s.reset_index(drop=True)
    
    #label and date removal
    clean_stg_lbl_1 = db_700s.columns[~pd.Series(db_700s.columns).str.startswith('Lbl_')]
    clean_stg_lbl_2 = clean_stg_lbl_1[~pd.Series(clean_stg_lbl_1).str.startswith('date_')]
    
    #below detection limit removal
    # dc_exclude = dc[dc['Primary_ID'].notnull()]
    # dc_exclude = dc_exclude.reset_index(drop=True)
    clean_stg_detection = clean_stg_lbl_2[~clean_stg_lbl_2.isin(dc_exclude['Primary_ID'])]
    db_clean = db_700s[clean_stg_detection][db_700s[clean_stg_detection][Intel_num].notnull()]
    db_clean = db_clean.reset_index(drop=True)
    
    return db_clean, Intel_num

models = {
    'ALL_QC': ['qc_bp_','qc_cp','cp1_sqc_','cp2_sqc_','bp_sqc_','end_qc_', 'qc_bot_','stg_cp1_bp','stg_cp2_bp','stg_bp_pfg'],
    'PP_RM': ['pp_','rm_','stg_cp1_bp','stg_cp2_bp','stg_bp_pfg'],
    'ALL_QC_until_bulk': ['qc_bp_','qc_cp','cp1_sqc_','cp2_sqc_','bp_sqc_','stg_cp1_bp','stg_cp2_bp'],
    'PP_RM_until_bulk': ['pp_bp_','pp_cp','rm_bp_','rm_cp','stg_cp1_bp','stg_cp2_bp']
}

import os

def build_models(db_clean, Intel_num, project_path):
    # db_clean, Intel_num = intel_cleaning(Product_name)
    
    scaler_dir = os.path.join(project_path, 'modelling', 'scalers')
    pls_dir = os.path.join(project_path, 'modelling', 'models', 'PLS')
    rf_dir = os.path.join(project_path, 'modelling', 'models', 'RF')
    xgb_dir = os.path.join(project_path, 'modelling', 'models', 'XGB')
    graphics_pls_dir = os.path.join(project_path, 'modelling', 'graphics', 'PLS')
    graphics_rf_dir = os.path.join(project_path, 'modelling', 'graphics', 'RF')
    graphics_xgb_dir = os.path.join(project_path, 'modelling', 'graphics', 'XGB')
    excel_pls_dir = os.path.join(project_path, 'modelling', 'excel', 'PLS')
    excel_rf_dir = os.path.join(project_path, 'modelling', 'excel', 'RF')
    excel_xgb_dir = os.path.join(project_path, 'modelling', 'excel', 'XGB')
    data_dir = os.path.join(project_path, 'modelling', 'data')
    for directory in [pls_dir, rf_dir, xgb_dir, graphics_pls_dir, graphics_rf_dir, graphics_xgb_dir, excel_pls_dir, excel_rf_dir, excel_xgb_dir, data_dir, scaler_dir]:
        print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

    #identifying x & y variable
    db_y = db_clean[Intel_num]
    db_x_init = db_clean[db_clean.columns[~(pd.Series(db_clean.columns).str.startswith('Intel_'))]].copy()
    
    #removing empty columns
    for column in db_x_init.columns:
        if db_x_init[column].isnull().sum() >= db_x_init.shape[0]*0.5 :
            db_x_init.drop([column], axis=1, inplace=True)
        else :
            db_x_init[column].fillna(db_x_init[column].mean(), inplace=True)
            
    #split dataset to All QC and PP+RM
    list_ALL_QC = ['qc_bp_','qc_cp','cp1_sqc_','cp2_sqc_','bp_sqc_','end_qc_', 'qc_bot_','stg_cp1_bp','stg_cp2_bp','stg_bp_pfg']
    list_PP_RM = ['pp_','rm_','stg_cp1_bp','stg_cp2_bp','stg_bp_pfg']
    list_ALL_QC_until_bulk = ['qc_bp_','qc_cp','cp1_sqc_','cp2_sqc_','bp_sqc_','stg_cp1_bp','stg_cp2_bp']
    list_PP_RM_until_bulk = ['pp_bp_','pp_cp','rm_bp_','rm_cp','stg_cp1_bp','stg_cp2_bp']
    list_models = [list_ALL_QC,list_PP_RM,list_ALL_QC_until_bulk,list_PP_RM_until_bulk]
    name_models = ['ALL_QC','PP_RM','ALL_QC_until_bulk','PP_RM_until_bulk']
    
    column_models = {}
    
    for i in range(len(list_models)):
        column_models[name_models[i]] = []
        for j in range(len(list_models[i])):
            col = db_x_init.columns[pd.Series(db_x_init.columns).str.startswith(list_models[i][j])]
            column_models[name_models[i]].extend(col)   
            
    # X variables (features)
    X_ALL_QC = db_x_init[column_models['ALL_QC']].copy()
    X_PP_RM = db_x_init[column_models['PP_RM']].copy()
    X_ALL_QC_until_bulk = db_x_init[column_models['ALL_QC_until_bulk']].copy()
    X_PP_RM_until_bulk = db_x_init[column_models['PP_RM_until_bulk']].copy()

    X_ALL_QC.to_excel(os.path.join(data_dir, 'data_ALL_QC.xlsx'), index=False)
    X_PP_RM.to_excel(os.path.join(data_dir, 'data_PP_RM.xlsx'), index=False)
    X_ALL_QC_until_bulk.to_excel(os.path.join(data_dir, 'data_ALL_QC_until_bulk.xlsx'), index=False)
    X_PP_RM_until_bulk.to_excel(os.path.join(data_dir, 'data_PP_RM_until_bulk.xlsx'), index=False)
    
    # y variables (labels)
    y_intel = db_y.copy()
    y_intel.to_excel(os.path.join(data_dir, 'data_y.xlsx'), index=False)

    # Min Max Scaling
    s_y = MinMaxScaler()
    s_ALL_QC = MinMaxScaler()
    s_PP_RM = MinMaxScaler()
    s_ALL_QC_until_bulk = MinMaxScaler()
    s_PP_RM_until_bulk = MinMaxScaler()
    
    # scaler_X = [s_ALL_QC, s_PP_RM, s_ALL_QC_until_bulk, s_PP_RM_until_bulk]
    X_set = {
        'ALL_QC': {
            'scaler': s_ALL_QC,
            'data': X_ALL_QC,
        },
        'PP_RM': {
            'scaler': s_PP_RM,
            'data': X_PP_RM,
        },
        'ALL_QC_until_bulk': {
            'scaler': s_ALL_QC_until_bulk,
            'data': X_ALL_QC_until_bulk
        },
        'PP_RM_until_bulk': {
            'scaler': s_PP_RM_until_bulk,
            'data': X_PP_RM_until_bulk
        }
    }

    y_intel = y_intel.to_frame()
    s_y.fit(y_intel)
    y_scaled = s_y.transform(y_intel)
    y_intel = pd.DataFrame(y_scaled, columns = y_intel.columns)

    for mode in X_set:
        X_set[mode]['scaler'].fit(X_set[mode]['data'])
        X_scaled = X_set[mode]['scaler'].transform(X_set[mode]['data'])
        X_set[mode]['data'] = pd.DataFrame(X_scaled, columns = X_set[mode]['data'].columns)
        
    scaler_all_qc_filename = path.join(scaler_dir, "s_ALL_QC.pkl")
    scaler_pp_rm_filename = path.join(scaler_dir, "s_PP_RM.pkl")
    scaler_all_qc_until_bulk_filename = path.join(scaler_dir, "s_ALL_QC_until_bulk.pkl")
    scaler_pp_rm_until_bulk_filename = path.join(scaler_dir, "s_PP_RM_until_bulk.pkl")
    scaler_y_filename = path.join(scaler_dir, "s_y.pkl")

    pickle.dump(s_ALL_QC, open(scaler_all_qc_filename, 'wb'))
    pickle.dump(s_PP_RM, open(scaler_pp_rm_filename,'wb'))
    pickle.dump(s_ALL_QC_until_bulk, open(scaler_all_qc_until_bulk_filename, 'wb'))
    pickle.dump(s_PP_RM_until_bulk, open(scaler_pp_rm_until_bulk_filename,'wb'))
    pickle.dump(s_y, open(scaler_y_filename, 'wb'))
    
    #Hyperparameter tuning
    def hypertune_pls_regressor(X, y):    
        pls = PLSRegression()
        np.random.seed(42)
        param_grid = { 
            "n_components": lst_num
        }
        
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        param_search = GridSearchCV(estimator=pls, param_grid=param_grid, scoring='neg_mean_squared_error', 
                                    cv=inner_cv, n_jobs=None, return_train_score=True, verbose=1)
        param_search.fit(X, y.values.ravel())
        #print_results_pls_regressor(param_search)
        return param_search

    def hypertune_random_forest(X, y):   
        rf = RandomForestRegressor()
        np.random.seed(42)
        param_grid = { 
            "n_estimators": lst_num,
            "max_depth": [2, 3, 4, 5, 6, 8, 10]
        }
        
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        param_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='neg_mean_squared_error', 
                                    cv=inner_cv, n_jobs=None, return_train_score=True, verbose=1)
        param_search.fit(X, y.values.ravel())
        #print_results_random_forest(param_search)
        return param_search

    def hypertune_xgboost_regression(X, y):    
        xgb_model = xgb.XGBRegressor()
        np.random.seed(42)
        param_grid = {
            "n_estimators": lst_num,
            "max_depth": [2, 3, 4, 5, 6],
            "learning_rate": [0.1, 0.08, 0.05, 0.01, 0.005],
            "min_child_weight": [1, 2, 4],
            "colsample_bytree": [0.3, 0.5, 0.7, 0.9],
            "subsample": [0.6, 0.8, 0.9]
        }
        # for testing
        param_grid = {
            "n_estimators": lst_num,
            "max_depth": [2],
            "learning_rate": [0.05],
            "min_child_weight": [4],
            "colsample_bytree": [0.7],
            "subsample": [0.8]
        }
    
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        param_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', 
                                    cv=inner_cv, n_jobs=-1, return_train_score=True, verbose=10)
        param_search.fit(X, y.values.ravel())
        #print_results_xgboost_regression(param_search)
        return param_search
    
    #Test - Train split
    for mode in X_set:
        X_intel = X_set[mode]['data']
        lab_x = mode
        #if X_intel == X_ALL_QC:
            #lab_x = 'ALL_QC'
        #if X_intel == X_PP_RM:
            #lab_x = 'PP_RM'
        #if X_intel == X_ALL_QC_until_bulk:
            #lab_x = 'ALL_QC_until_bulk'
        #if X_intel == X_PP_RM_until_bulk:
            #lab_x = 'PP_RM_until_bulk'
            
        X_train, X_test, y_train, y_test = train_test_split(X_intel, y_intel, test_size=0.3, random_state=42)
        df_train = pd.concat([X_train, y_train], axis=1, sort=False)
        df_test = pd.concat([X_test, y_test], axis=1, sort=False)
        
        max_num = X_intel.shape[1]
        min_num = max_num/30
        array_num = np.linspace(min_num, max_num,1) # FOR TESTING. was 10 before
        lst_num = [int(i) for i in array_num]
        
        #Modelling
        param_search_pls = hypertune_pls_regressor(X_intel, y_intel)
        model_pls = PLSRegression(**param_search_pls.best_params_)

        np.random.seed(42)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores_pls = cross_validate(estimator=model_pls, X=X_intel, y=y_intel, cv=outer_cv, 
                                       scoring=["neg_mean_squared_error", "neg_root_mean_squared_error", "neg_median_absolute_error", "r2"])
        model_pls.fit(X_train, y_train.values.ravel())

        pkl_filename = path.join(pls_dir, str(lab_x)+'.pkl')
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model_pls, file)

###############################################################################################################    
        param_search_rf = hypertune_random_forest(X_intel, y_intel)
        model_rf = RandomForestRegressor(**param_search_rf.best_params_)

        np.random.seed(42)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores_rf = cross_validate(estimator=model_rf, X=X_intel, y=y_intel, cv=outer_cv, 
                                  scoring=["neg_mean_squared_error", "neg_root_mean_squared_error", "neg_median_absolute_error", "r2"])
        model_rf.fit(X_train, y_train.values.ravel())

        pkl_filename = path.join(rf_dir, str(lab_x)+'.pkl')
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model_rf, file)
    
################################################################################################################
        param_search_xgb = hypertune_xgboost_regression(X_intel, y_intel)
        model_xgb = xgb.XGBRegressor(**param_search_xgb.best_params_)

        np.random.seed(42)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores_xgb = cross_validate(estimator=model_xgb, X=X_intel, y=y_intel, cv=outer_cv, 
                                  scoring=["neg_mean_squared_error", "neg_root_mean_squared_error", "neg_median_absolute_error", "r2"])
        model_xgb.fit(X_train, y_train.values.ravel())

        pkl_filename = path.join(xgb_dir, str(lab_x)+'.pkl')
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model_xgb, file)
            
        model_lst = [model_pls, model_rf, model_xgb]
        cv_scores_lst = [cv_scores_pls, cv_scores_rf, cv_scores_xgb]
        tag = ['pls', 'rf', 'xgb']
        TAG = ['PLS', 'RF', 'XGB']
        base_paths = [graphics_pls_dir, graphics_rf_dir, graphics_xgb_dir]
        
        for i in range(len(model_lst)):
            base_path = base_paths[i]

            df_train['y_pred_'+str(tag[i])] = model_lst[i].predict(X_train)
            df_train['error_'+str(tag[i])] = df_train[Intel_num] - df_train['y_pred_'+str(tag[i])]
            df_test['y_pred_'+str(tag[i])] = model_lst[i].predict(X_test)
            df_test['error_'+str(tag[i])] = df_test[Intel_num] - df_test['y_pred_'+str(tag[i])]
            
            score_r2_mean = np.mean(cv_scores_lst[i]['test_r2'])
            score_r2_std = np.std(cv_scores_lst[i]['test_r2'])
            score_r2_train = r2_score(df_train[Intel_num], df_train['y_pred_'+str(tag[i])])
            score_r2_test = r2_score(df_test[Intel_num], df_test['y_pred_'+str(tag[i])])
    
            score_rmse_mean = np.mean(np.sqrt(-cv_scores_lst[i]['test_neg_mean_squared_error']))
            score_rmse_std = np.std(np.sqrt(-cv_scores_lst[i]['test_neg_mean_squared_error']))
            score_rmse_train = np.sqrt(mean_squared_error(df_train[Intel_num], df_train['y_pred_'+str(tag[i])]))
            score_rmse_test = np.sqrt(mean_squared_error(df_test[Intel_num], df_test['y_pred_'+str(tag[i])]))
    
            # Predicted vs True values
            fig, axs = plt.subplots(figsize=(5,5))
            sns.scatterplot(x=Intel_num, y='y_pred_'+str(tag[i]), data=df_train, ax=axs)
            axs.set(xlim=(0,1))
            axs.set(ylim=(0,1))
            axs.set_title(str(TAG[i])+' Prediction (Train Set) - '+lab_x+'_'+ str(Intel_num))
            axs.set(xlabel= str(Intel_num)+' (observed)')
            axs.set(ylabel= str(Intel_num)+' (predicted)')
            axs.text(0.1, 0.9, 'Train R2: '+'{:.2f}'.format(score_r2_train), transform = axs.transAxes)
            axs.text(0.1, 0.8, 'CV R2: '+'{:.2f}'.format(score_r2_mean) + ' +/- '+'{:.2f}'.format(score_r2_std), transform = axs.transAxes)
            plt.savefig(path.join(base_path, str(TAG[i])+' Prediction (Train Set) - '+lab_x+'_'+str(Intel_num)+'.png'))
            # Errors vs True values
            fig, axs = plt.subplots(figsize=(5,5))
            sns.scatterplot(x=Intel_num, y='error_'+str(tag[i]), data=df_train, ax=axs)
            axs.set(xlim=(0,1))
            axs.set(ylim=(-0.5,0.5))
            axs.set_title(str(TAG[i])+' Residuals (Train Set) - '+lab_x+'_'+str(Intel_num))
            axs.set(xlabel= str(Intel_num)+' (observed)')
            axs.set(ylabel='error in '+str(Intel_num)+' (predicted)')
            axs.text(0.1, 0.9, 'Train RMSE: '+'{:.2f}'.format(score_rmse_train), transform = axs.transAxes)
            axs.text(0.1, 0.8, 'CV RMSE: '+'{:.2f}'.format(score_rmse_mean) + ' +/- '+'{:.2f}'.format(score_rmse_std), transform = axs.transAxes)
            plt.savefig(path.join(base_path, str(TAG[i])+' Residuals (Train Set) - '+lab_x+'_'+str(Intel_num)+'.png'))
            # Predicted vs True values
            fig, axs = plt.subplots(figsize=(5,5))
            sns.scatterplot(x=Intel_num, y='y_pred_'+str(tag[i]), data=df_test, ax=axs)
            axs.set(xlim=(0,1))
            axs.set(ylim=(0,1))
            axs.set_title(str(TAG[i])+' Prediction (Test Set) - '+lab_x+'_'+str(Intel_num))
            axs.set(xlabel= str(Intel_num)+' (observed)')
            axs.set(ylabel= str(Intel_num)+' (predicted)')
            axs.text(0.1, 0.9, 'Test R2: '+'{:.2f}'.format(score_r2_test), transform = axs.transAxes)
            axs.text(0.1, 0.8, 'CV R2: '+'{:.2f}'.format(score_r2_mean) + ' +/- '+'{:.2f}'.format(score_r2_std), transform = axs.transAxes)
            plt.savefig(path.join(base_path, str(TAG[i])+' Prediction (Test Set) - '+lab_x+'_'+str(Intel_num)+'.png'))
            # Errors vs True values
            fig, axs = plt.subplots(figsize=(5,5))
            sns.scatterplot(x=Intel_num, y='error_'+str(tag[i]), data=df_test, ax=axs)
            axs.set(xlim=(0,1))
            axs.set(ylim=(-0.5,0.5))
            axs.set_title(str(TAG[i])+' Residuals (Test Set) - '+lab_x+'_'+str(Intel_num))
            axs.set(xlabel= str(Intel_num)+' (observed)')
            axs.set(ylabel='error in '+str(Intel_num)+' (predicted)')
            axs.text(0.1, 0.9, 'Test RMSE: '+'{:.2f}'.format(score_rmse_test), transform = axs.transAxes)
            axs.text(0.1, 0.8, 'CV RMSE: '+'{:.2f}'.format(score_rmse_mean) + ' +/- '+'{:.2f}'.format(score_rmse_std), transform = axs.transAxes)
            plt.savefig(path.join(base_path, str(TAG[i])+' Residuals (Test Set) - '+lab_x+'_'+str(Intel_num)+'.png'))

if __name__ == '__main__':
    ##load data
    int_file_dataset = 'C:/Users/M304498/Intel Project/test_data/PHPS_DATA_SET_P2_Intel20211027.xlsx'
    int_sheet_dataset = 'Dataset'
    int_file_exclude = 'C:/Users/M304498/Intel Project/test_data/exclude_smaller.xlsx'
    int_sheet_exclude = 'Tabelle1'
    Product_name = 'SP710'

    print('reading dataset')
    dataset = pd.read_excel(int_file_dataset, sheet_name = int_sheet_dataset)
    print('reading exclude')
    exclude = pd.read_excel(int_file_exclude, sheet_name = int_sheet_exclude)

    print(dataset)
    print(exclude)

    print('preprocessing dataset')
    dataset = prepare_main_dataset(dataset)
    print('preprocessing exclude')
    exclude = prepare_exclude_dataset(exclude)

    print('cleaning dataset')
    cleaned, column = clean_dataset('SP710', dataset, exclude)
    print('making models')
    build_models(cleaned, column, path.join('C:\\', 'Users', 'M304498', 'test_project'))
