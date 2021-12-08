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
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error

##load data
#tsmc_file_dataset = './TSMC_Master_dataset/Data_SA_02.xlsx'
#tsmc_sheet_dataset = 'FG_ECD'
#col_x = 5
#label = 'co_sel_ecd'

#def tsmc_main_dataset_loading(tsmc_file_dataset, tsmc_sheet_dataset):
    #db = pd.read_excel(tsmc_file_dataset, sheet_name = tsmc_sheet_dataset)
    #return db

def intel_model_building(df, col_x, label):
    # db = tsmc_main_dataset_loading(tsmc_file_dataset, tsmc_sheet_dataset)
    db = df
    
    #identifying x & y variable
    db_x = db.iloc[:,col_x:]
    db_y = db.loc[:,label].copy()
    db_y_base = db.loc[:,label].copy()
    
    #removing empty x columns
    for column in db_x.columns:
        if db_x[column].isnull().sum() >= db_x.shape[0]*0.5 :
            db_x.drop([column], axis=1, inplace=True)
        else :
            db_x[column].fillna(db_x[column].mean(), inplace=True)
            
    #removing empty y columns
    for i in range(db_y_base.shape[0]):
        if pd.isnull(db_y_base[i]) == True:
            db_x.drop(index = i, inplace=True)
            db_y.drop(index = i, inplace=True)
        else:
            pass
    
    db_x = db_x.reset_index(drop=True)
    db_y = db_y.reset_index(drop=True)
    
    # X variables (features)
    X = db_x.copy()
    
    #X.to_excel('data_X.xlsx', index=False)

    # y variables (labels)
    y = db_y.copy()

    # Min Max Scaling
    s_x = MinMaxScaler()
    s_y = MinMaxScaler()

    s_x.fit(X)
    X_scaled = s_x.transform(X)
    X = pd.DataFrame(X_scaled, columns = X.columns)

    y = y.to_frame()
    s_y.fit(y)
    y_scaled = s_y.transform(y)
    y = pd.DataFrame(y_scaled, columns = y.columns)

    scaler_x_filename = "s_x"+str(label)+".pkl"
    scaler_y_filename = "s_y"+str(label)+".pkl"

    #pickle.dump(s_x, open(scaler_x_filename, 'wb'))
    #pickle.dump(s_y, open(scaler_y_filename, 'wb'))
    
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
        print_results_pls_regressor(param_search)
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
        print_results_random_forest(param_search)
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
    
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        param_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', 
                                    cv=inner_cv, n_jobs=-1, return_train_score=True, verbose=10)
        param_search.fit(X, y.values.ravel())
        print_results_xgboost_regression(param_search)
        return param_search
    
    #Test - Train split            
    X_train, X_test, y_train, y_test = train_test_split(X_intel, y_intel, test_size=0.3, random_state=42)
    df_train = pd.concat([X_train, y_train], axis=1, sort=False)
    df_test = pd.concat([X_test, y_test], axis=1, sort=False)
        
    max_num = X_intel.shape[1]
    min_num = max_num/30
    array_num = np.linspace(min_num, max_num,10)
    lst_num = [int(i) for i in array_num]
        
    #Modelling
    param_search_pls = hypertune_pls_regressor(X_intel, y_intel)
    model_pls = PLSRegression(**param_search_pls.best_params_)

    np.random.seed(42)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_pls = cross_validate(estimator=model_pls, X=X_intel, y=y_intel, cv=outer_cv, 
                                   scoring=["neg_mean_squared_error", "neg_root_mean_squared_error", "neg_median_absolute_error", "r2"])
    model_pls.fit(X_train, y_train.values.ravel())

    pkl_filename = 'pls_model_'+str(lab_x)+str(label)'.pkl'
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

    pkl_filename = 'rf_model_'+str(lab_x)+str(label)'.pkl'
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

    pkl_filename = 'xgb_model_'+str(lab_x)+str(label)'.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model_xgb, file)
            
    model_lst = [model_pls, model_rf, model_xgb]
    cv_scores_lst = [cv_scores_pls, cv_scores_rf, cv_scores_xgb]
    tag = ['pls', 'rf', 'xgb']
    TAG = ['PLS', 'RF', 'XGB']
        
    for i range(len(model_lst)):
        df_train['y_pred_'+str(tag[i])] = model_lst[i].predict(X_train)
        df_train['error_'+str(tag[i])] = df_train[label] - df_train['y_pred_'+str(tag[i])]
        df_test['y_pred_'+str(tag[i])] = model_lst[i].predict(X_test)
        df_test['error_'+str(tag[i])] = df_test[label] - df_test['y_pred_'+str(tag[i])]
            
        score_r2_mean = np.mean(cv_scores_lst[i]['test_r2'])
        score_r2_std = np.std(cv_scores_lst[i]['test_r2'])
        score_r2_train = r2_score(df_train[label], df_train['y_pred_'+str(tag[i])])
        score_r2_test = r2_score(df_test[label], df_test['y_pred_'+str(tag[i])])
    
        score_rmse_mean = np.mean(np.sqrt(-cv_scores_lst[i]['test_neg_mean_squared_error']))
        score_rmse_std = np.std(np.sqrt(-cv_scores_lst[i]['test_neg_mean_squared_error']))
        score_rmse_train = np.sqrt(mean_squared_error(df_train[label], df_train['y_pred_'+str(tag[i])]))
        score_rmse_test = np.sqrt(mean_squared_error(df_test[label], df_test['y_pred_'+str(tag[i])]))
    
        # Predicted vs True values
        fig, axs = plt.subplots(figsize=(5,5))
        sns.scatterplot(x=label, y='y_pred_'+str(tag[i]), data=df_train, ax=axs)
        axs.set(xlim=(0,1))
        axs.set(ylim=(0,1))
        axs.set_title(str(TAG[i])+' Prediction (Train Set) - '+ str(label))
        axs.set(xlabel= str(label)+' (observed)')
        axs.set(ylabel= str(label)+' (predicted)')
        axs.text(0.1, 0.9, 'Train R2: '+'{:.2f}'.format(score_r2_train), transform = axs.transAxes)
        axs.text(0.1, 0.8, 'CV R2: '+'{:.2f}'.format(score_r2_mean) + ' +/- '+'{:.2f}'.format(score_r2_std), transform = axs.transAxes)
        plt.savefig(str(TAG[i])+' Prediction (Train Set) - '+str(label)+'.jpeg')
        # Errors vs True values
        fig, axs = plt.subplots(figsize=(5,5))
        sns.scatterplot(x=label, y='error_pls_1', data=df_train, ax=axs)
        axs.set(xlim=(0,1))
        axs.set(ylim=(-0.5,0.5))
        axs.set_title(str(TAG[i])+' Residuals (Train Set) - '+str(label))
        axs.set(xlabel= str(label)+' (observed)')
        axs.set(ylabel='error in '+str(label)+' (predicted)')
        axs.text(0.1, 0.9, 'Train RMSE: '+'{:.2f}'.format(score_rmse_train), transform = axs.transAxes)
        axs.text(0.1, 0.8, 'CV RMSE: '+'{:.2f}'.format(score_rmse_mean) + ' +/- '+'{:.2f}'.format(score_rmse_std), transform = axs.transAxes)
        plt.savefig(str(TAG[i])+' Residuals (Train Set) - '+str(label)+'.jpeg')
        # Predicted vs True values
        fig, axs = plt.subplots(figsize=(5,5))
        sns.scatterplot(x=label, y='y_pred_'+str(tag[i]), data=df_test, ax=axs)
        axs.set(xlim=(0,1))
        axs.set(ylim=(0,1))
        axs.set_title(str(TAG[i])+' Prediction (Test Set) - '+str(label))
        axs.set(xlabel= str(label)+' (observed)')
        axs.set(ylabel= str(label)+' (predicted)')
        axs.text(0.1, 0.9, 'Test R2: '+'{:.2f}'.format(score_r2_test), transform = axs.transAxes)
        axs.text(0.1, 0.8, 'CV R2: '+'{:.2f}'.format(score_r2_mean) + ' +/- '+'{:.2f}'.format(score_r2_std), transform = axs.transAxes)
        plt.savefig(str(TAG[i])+' Prediction (Test Set) - '+str(label)+'.jpeg')
        # Errors vs True values
        fig, axs = plt.subplots(figsize=(5,5))
        sns.scatterplot(x=label, y='error_pls_1', data=df_test, ax=axs)
        axs.set(xlim=(0,1))
        axs.set(ylim=(-0.5,0.5))
        axs.set_title(str(TAG[i])+' Residuals (Test Set) - '+str(label))
        axs.set(xlabel= str(label)+' (observed)')
        axs.set(ylabel='error in '+str(label)+' (predicted)')
        axs.text(0.1, 0.9, 'Test RMSE: '+'{:.2f}'.format(score_rmse_test), transform = axs.transAxes)
        axs.text(0.1, 0.8, 'CV RMSE: '+'{:.2f}'.format(score_rmse_mean) + ' +/- '+'{:.2f}'.format(score_rmse_std), transform = axs.transAxes)
        plt.savefig(str(TAG[i])+' Residuals (Test Set) - '+str(label)+'.jpeg')


if __name__ == '__main__':
    tsmc_file_dataset = 'test_data/TSMC_Master_dataset/Data_SA_02.xlsx'
    tsmc_sheet_dataset = 'FG_ECD'
    col_x = 5
    label = 'co_sel_ecd'

    df = pd.read_excel(tsmc_file_dataset, sheet_name = tsmc_sheet_dataset)
    build_model(df, col_x, label)
