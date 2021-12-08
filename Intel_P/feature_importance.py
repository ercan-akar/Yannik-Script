# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
#import xlsxwriter
import pickle
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


## DATA IMPORT AND DATA PREPARATION

#df_initial = pd.read_excel('Peak_dataset_fid_4.xlsx', sheet_name = 'Sheet1', header = 0)

#df = df_initial.copy()

## Split complete data frame into 3 data sets

#df_X = df.iloc[0:, 5:36]
#df_Y1 = df.iloc[0:, 4]
#df_Y2 = df.iloc[0:, 4]
#df_Y3 = df.iloc[0:, 4]

## Fill empty values
#for column in df_X.iloc[:,:].columns:
    #if df_X[column].isnull().sum() >= (df_X.shape[0] * 0.5):
        #df_X.drop([column], axis=1, inplace=True)
    #else:
        #df_X[column].fillna(df_X[column].mean(), inplace=True)
        
## Min max scaling

#scaler = MinMaxScaler()

#scaler.fit(df_X)
#df_X_scaled = scaler.transform(df_X)
#DF_X = pd.DataFrame(df_X_scaled, columns = df_X.columns)

#df_Y1 = df_Y1.to_frame()
#scaler.fit(df_Y1)
#df_Y1_scaled = scaler.transform(df_Y1)
#DF_Y1 = pd.DataFrame(df_Y1_scaled, columns = df_Y1.columns)

#df_Y2 = df_Y2.to_frame()
#scaler.fit(df_Y2)
#df_Y2_scaled = scaler.transform(df_Y2)
#DF_Y2 = pd.DataFrame(df_Y2_scaled, columns = df_Y2.columns)

#df_Y3 = df_Y3.to_frame()
#scaler.fit(df_Y3)
#df_Y3_scaled = scaler.transform(df_Y3)
#DF_Y3 = pd.DataFrame(df_Y3_scaled, columns = df_Y3.columns)






### Import the pickles (it is necessary to import pickle)

#file_rf = 'Random_Forest'
#file_xgb = 'XGBoost'
#file_pls = 'Partial_Least_Squares'

#infile_rf = open(file_rf, 'rb')
#model_rf = pickle.load(infile_rf)
#infile_rf.close()

#infile_xgb = open(file_xgb, 'rb')
#model_xgb = pickle.load(infile_xgb)
#infile_xgb.close()

#infile_pls = open(file_pls, 'rb')
#model_pls = pickle.load(infile_pls)
#infile_pls.close()

##################### Print Feature importance
#
#model_rf.fit(DF_X, DF_Y1)
#model_xgb.fit(DF_X, DF_Y1)
#model_pls.fit(DF_X, DF_Y1)

# assess permutation importance
def compute_feature_importance(path):
    num_permutations = 25
    check_path    = os.path.join(path, 'feature_importance', 'check')
    graphics_path = os.path.join(path, 'feature_importance', 'graphics')
    vips_path     = os.path.join(path, 'feature_importance', 'vips')


    for directory in [check_path, graphics_path, vips_path]:
        print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

    modes = ['ALL_QC', 'PP_RM', 'ALL_QC_until_bulk', 'PP_RM_until_bulk']

    DF_Y1 = pd.read_excel(os.path.join(path, 'modelling', 'data', 'data_y.xlsx'))
    s_y = pickle.load(open(os.path.join(path, 'modelling', 'scalers', 's_y.pkl'), 'rb'))
    DF_Y1_s = s_y.transform(DF_Y1)
    DF_Y1 = pd.DataFrame(DF_Y1_s, columns = DF_Y1.columns)

    for mode in modes:
        model_rf  = pickle.load(open(os.path.join(path, 'modelling', 'models', 'RF',  '{}.pkl'.format(mode)), 'rb'))
        model_pls = pickle.load(open(os.path.join(path, 'modelling', 'models', 'PLS', '{}.pkl'.format(mode)), 'rb'))
        model_xgb = pickle.load(open(os.path.join(path, 'modelling', 'models', 'XGB', '{}.pkl'.format(mode)), 'rb'))

        DF_X = pd.read_excel(os.path.join(path, 'modelling', 'data', 'data_{}.xlsx'.format(mode)))
        s = pickle.load(open(os.path.join(path, 'modelling', 'scalers', 's_{}.pkl'.format(mode)), 'rb'))
        DF_X_s = s.transform(DF_X)
        DF_X = pd.DataFrame(DF_X_s, columns = DF_X.columns)


        importance_perm_rf = permutation_importance(
            model_rf,
            DF_X,
            DF_Y1,
            n_repeats=num_permutations,
            random_state=42,
            n_jobs=None
        )

        importance_perm_xgb = permutation_importance(
            model_xgb,
            DF_X,
            DF_Y1,
            n_repeats=num_permutations,
            random_state=42,
            n_jobs=-1
        )

        importance_perm_pls = permutation_importance(
            model_pls,
            DF_X,
            DF_Y1,
            n_repeats=num_permutations,
            random_state=42,
            n_jobs=None
        )

        num_top_features = 20
        num_selected_features = 10

        #Check 25 iterations


        checkrf1 = pd.DataFrame(importance_perm_rf.importances)
        checkxgb1 = pd.DataFrame(importance_perm_xgb.importances)
        checkpls1 = pd.DataFrame(importance_perm_pls.importances)

        checkrf1.index = DF_X.columns
        checkxgb1.index = DF_X.columns
        checkpls1.index = DF_X.columns

        checkrf1.to_excel(os.path.join(check_path, 'data_rf_{}.xlsx'.format(mode)))
        checkxgb1.to_excel(os.path.join(check_path, 'data_xgb_{}.xlsx'.format(mode)))
        checkpls1.to_excel(os.path.join(check_path, 'data_pls_{}.xlsx'.format(mode)))

        # VIP lists

        VIP_list_rf1 = []
        VIP_list_xgb1 = []
        VIP_list_pls1 = []

        # VIP functions

        def list_VIP(VIP_1):
            VIP_2 = pd.DataFrame(VIP_1).T
            VIP_3 = VIP_2.reindex(index=VIP_2.index[::-1])
            VIP_4 = VIP_3.iloc[0:num_top_features]
            VIP_5 = VIP_4.reset_index(drop=True)
            return(VIP_5)

        # Random Forest =============================================================

        importance_results = importance_perm_rf
        fig, ax = plt.subplots(figsize=(5,5))

        sorted_idx = importance_results.importances_mean.argsort()
        VIP_list_rf1.append(DF_X.columns[sorted_idx])

        ax.boxplot(
            x=importance_results.importances[sorted_idx][-num_top_features:].T,
            labels=DF_X.columns[sorted_idx][-num_top_features:],
            vert=False
        )

        ax.set_title("Permutation Importance (Random Forest, {})".format(mode))
        fig.tight_layout()

        #plt.show()
        top_features_rf = DF_X.columns[sorted_idx][-num_selected_features:]

        #save_results_to = '/Users/M304594/Desktop/Results/'

        fig.savefig(os.path.join(graphics_path, 'Permutation Importance (Random Forest, {}).png'.format(mode)))

        VIP_list_rf = list_VIP(VIP_list_rf1)
        VIP_list_rf.to_excel(os.path.join(vips_path, 'VIP_list_rf_{}.xlsx'.format(mode), header=False, index=False))

        # XGBoost ===================================================================

        importance_results = importance_perm_xgb
        fig, ax = plt.subplots(figsize=(5,5))

        sorted_idx = importance_results.importances_mean.argsort()
        VIP_list_xgb1.append(DF_X.columns[sorted_idx])

        ax.boxplot(
            x=importance_results.importances[sorted_idx][-num_top_features:].T,
            labels=DF_X.columns[sorted_idx][-num_top_features:],
            vert=False
        )

        ax.set_title("Permutation Importance (XGBoost, {})".format(mode))
        fig.tight_layout()

        #plt.show()
        top_features_xgb = DF_X.columns[sorted_idx][-num_selected_features:]

        #save_results_to = '/Users/???/Desktop/Results/'

        # fig.savefig(save_results_to + '/XGBoost/Permutation Importance (XGBoost) - (Customer_data).png')
        fig.savefig(os.path.join(graphics_path, 'Permutation Importance (XGBoost, {}).png'.format(mode)))

        VIP_list_xgb = list_VIP(VIP_list_xgb1)
        VIP_list_xgb.to_excel(os.path.join(vips_path, 'VIP_list_xgb_{}.xlsx'.format(mode), header=False, index=False))

        # PLS ===================================================================

        importance_results = importance_perm_pls
        fig, ax = plt.subplots(figsize=(5,5))

        sorted_idx = importance_results.importances_mean.argsort()
        VIP_list_pls1.append(DF_X.columns[sorted_idx])

        ax.boxplot(
            x=importance_results.importances[sorted_idx][-num_top_features:].T,
            labels=DF_X.columns[sorted_idx][-num_top_features:],
            vert=False
        )

        ax.set_title("Permutation Importance (PLS, {})".format(mode))
        fig.tight_layout()

        #plt.show()
        top_features_pls = DF_X.columns[sorted_idx][-num_selected_features:]

        #save_results_to = '/Users/???/Desktop/Results/'

        #fig.savefig(save_results_to + '/Partial Least Squares/Permutation Importance (PLS) - (Customer_data).png')
        fig.savefig(os.path.join(graphics_path, 'Permutation Importance (PLS, {}).png'.format(mode)))

        VIP_list_pls = list_VIP(VIP_list_pls1)
        VIP_list_pls.to_excel(os.path.join(vips_path, 'VIP_list_pls_{}.xlsx'.format(mode), header=False, index=False))

        ###### Merge results of VIP lists

        scores = {'value':  [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}
        sc = pd.DataFrame(scores)
        weigths = importance_results.importances_mean
        wt = pd.DataFrame(weigths).sort_values(0, ascending=False)

        VIP_rf_sc = pd.concat([VIP_list_rf, sc], axis=1)
        VIP_xgb_sc = pd.concat([VIP_list_xgb, sc], axis=1)
        VIP_pls_sc = pd.concat([VIP_list_pls, sc], axis=1)

        VIP_rf_wt = pd.concat([VIP_list_rf, wt], axis=1)
        VIP_xgb_wt = pd.concat([VIP_list_xgb, wt], axis=1)
        VIP_pls_wt = pd.concat([VIP_list_pls, wt], axis=1)

        VIP_premerged_sc = VIP_rf_sc.groupby(0).sum().add(VIP_xgb_sc.groupby(0).sum(), fill_value=0).reset_index()
        VIP_merged_sc = VIP_premerged_sc.groupby(0).sum().add(VIP_pls_sc.groupby(0).sum(), fill_value=0).reset_index()
        VIP_sorted_sc = VIP_merged_sc.sort_values('value', ascending=False) # TODO save this

        VIP_premerged_wt = VIP_rf_wt.groupby(0).sum().add(VIP_xgb_wt.groupby(0).sum(), fill_value=0).reset_index()
        VIP_merged_wt = VIP_premerged_wt.groupby(0).sum().add(VIP_pls_wt.groupby(0).sum(), fill_value=0).reset_index()
        VIP_sorted_wt = VIP_merged_wt.sort_values('value', ascending=False) # TODO save this

        VIP_sorted_sc.to_excel(os.path.join(vips_path, 'VIP_sorted_sc_{}.xlsx'.format(mode)), index = False)
        VIP_sorted_wt.to_excel(os.path.join(vips_path, 'VIP_sorted_wt_{}.xlsx'.format(mode)), index = False)
