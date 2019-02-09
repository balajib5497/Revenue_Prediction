import numpy as np
import pandas as pd

import datetime as dt
import json
import time
import gc
gc.enable()

from fiscalyear import FiscalDate

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import KFold, TimeSeriesSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix

from util import *

import lightgbm as lgb
import xgboost as xgb

from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize, dump, load
from skopt.plots import plot_convergence

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SMOTENC
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks, RepeatedEditedNearestNeighbours
from imblearn.combine import SMOTEENN

import pickle

#pd.options.display.max_rows = 500
#pd.options.display.max_columns = 500

print('reading data & json blobs cleaning ...')

#train = pd.read_csv('../dataset/train.csv')
#train = pd.io.parsers.read_csv('../dataset/train.csv', iterator = True, chunksize = 1000)
#train = pd.concat(train, ignore_index=True)
#train = clean_json_blobs(train, is_train=True)

#test = pd.read_csv('../dataset/test.csv')
#test = pd.io.parsers.read_csv('../dataset/test.csv', iterator = True, chunksize = 1000)
#test = pd.concat(test, ignore_index=True)
#test = clean_json_blobs(test)

#train.to_csv('./train_p.csv', index=False)
#test.to_csv('./test_p.csv', index=False)

train = pd.read_csv('./train_p.csv')
#train = pd.io.parsers.read_csv('./train_p.csv', iterator = True, chunksize = 1000)
#train = pd.concat(train, ignore_index=True)
    
test = pd.read_csv('./test_p.csv')
#test = pd.io.parsers.read_csv('./test_p.csv', iterator = True, chunksize = 1000)
#test = pd.concat(test, ignore_index=True)

print('starts preprocessing ...')

#train = train[~train.duplicated(subset=['sessionId'])]

train.sort_values(by='visitStartTime', inplace=True)
test.sort_values(by='visitStartTime', inplace=True)

train['fullVisitorId'] = train['sessionId'].apply(lambda x:x.split('_')[0])
train_full_visitor_id = train['fullVisitorId']

test['fullVisitorId'] = test['sessionId'].apply(lambda x:x.split('_')[0])
test_full_visitor_id = test['fullVisitorId']

sample_submission = pd.read_csv('../dataset/sample_submission.csv')
sample_submission_ids = sample_submission['fullVisitorId'].to_dict()
sample_submission_ids = {int(value):key for key, value in sample_submission_ids.items()}
test_extra_index = [i for i in train_full_visitor_id.index if int(train_full_visitor_id[i]) in sample_submission_ids]

for col in (cols_to_fill_zero + ['totals_transactionRevenue']):
    train[col] = train[col].astype(float)
    train[col].fillna(value=0, inplace=True)
    
# train['traffic_adclickinfo_isnull'] = train['traffic_gclId'].apply(lambda x: 1.0 if type(x) != str and math.isnan(x) else 0.0)
train.drop(columns=cols_to_drop, axis=1, inplace=True)  

#train['traffic_keyword'] = train['traffic_keyword'].apply(token_normalization)
#train['traffic_keyword_count'] = train.groupby('traffic_keyword')['sessionId'].transform('nunique')
#train['traffic_keyword_count_unique'] = train.groupby('traffic_keyword')['fullVisitorId'].transform('nunique')
#
#train['traffic_keyword_count'].fillna(value=0, inplace=True)
#train['traffic_keyword_count_unique'].fillna(value=0, inplace=True)
#    
#test['traffic_keyword'] = test['traffic_keyword'].apply(token_normalization)
#test['traffic_keyword_count'] = test.groupby('traffic_keyword')['sessionId'].transform('nunique')
#test['traffic_keyword_count_unique'] = test.groupby('traffic_keyword')['fullVisitorId'].transform('nunique')
#
#test['traffic_keyword_count'].fillna(value=0, inplace=True)
#test['traffic_keyword_count_unique'].fillna(value=0, inplace=True)

#add_time_period_of_same_ID(train)
#add_time_period_of_same_ID(test)

train['totals_hits'] = train['totals_hits'].astype(float)
train['visitStartTime'] = train['visitStartTime'].apply(lambda x:dt.datetime.fromtimestamp(x))
train['visit_date'] = pd.to_datetime(train['visitStartTime'].apply(lambda x:x.date()))
train['visit_time'] = train['visitStartTime'].apply(lambda x:x.time())
train.drop('visitStartTime', axis=1, inplace=True)

train['date'] = train['date'].apply(to_date)
train['date_diff'] = (train['visit_date'] - train['date']).apply(lambda x: int(str(x)[0]))
train['visit_date'] = train['visit_date'].apply(lambda x:x.date())
date_col = train['date'].copy()
train.drop('date', axis=1, inplace=True)

#train['device_browser'] = train['device_browser'].apply(eliminate_low_freq_browsers)
#train['device_browser'] = train['device_browser'].apply(browser_mapping)
#train['traffic_adContent'] = train['traffic_adContent'].apply(adcontents_mapping)
#train['traffic_source'] = train['traffic_source'].apply(source_mapping)
#train['traffic_campaign'] = train['traffic_campaign'].apply(campaign_mapping)
#train['traffic_keyword'] = train['traffic_keyword'].apply(token_normalization)


train['traffic_isTrueDirect'] = train['traffic_isTrueDirect'].apply(value_or_null, args=(True, ))
train['traffic_adclick_isVideoAd'] = train['traffic_adclick_isVideoAd'].apply(value_or_null, args=(False, ))

for col in cols_to_fill_zero:
    test[col] = test[col].astype(float)
    test[col].fillna(value=0, inplace=True)

# test['traffic_adclickinfo_isnull'] = test['traffic_gclId'].apply(lambda x: 1.0 if type(x) != str and math.isnan(x) else 0.0)
test.drop(columns=cols_to_drop, axis=1, inplace=True)  

test['totals_hits'] = test['totals_hits'].astype(float)
test['visitStartTime'] = test['visitStartTime'].apply(lambda x:dt.datetime.fromtimestamp(x))
test['visit_date'] = pd.to_datetime(test['visitStartTime'].apply(lambda x:x.date()))
test['visit_time'] = test['visitStartTime'].apply(lambda x:x.time())
test.drop('visitStartTime', axis=1, inplace=True)

test['date'] = test['date'].apply(to_date)
test['date_diff'] = (test['visit_date'] - test['date']).apply(lambda x: int(str(x)[0]))
test['visit_date'] = test['visit_date'].apply(lambda x:x.date())
test.drop('date', axis=1, inplace=True)

#test['device_browser'] = test['device_browser'].apply(eliminate_low_freq_browsers)
#test['device_browser'] = test['device_browser'].apply(browser_mapping)
#test['traffic_adContent'] = test['traffic_adContent'].apply(adcontents_mapping)
#test['traffic_source'] = test['traffic_source'].apply(source_mapping)
#test['traffic_campaign'] = test['traffic_campaign'].apply(campaign_mapping)
#test['traffic_keyword'] = test['traffic_keyword'].apply(token_normalization)

test['traffic_isTrueDirect'] = test['traffic_isTrueDirect'].apply(value_or_null, args=(True, ))
test['traffic_adclick_isVideoAd'] = test['traffic_adclick_isVideoAd'].apply(value_or_null, args=(False, ))

train['day'] = train['visit_date'].apply(lambda x:x.day)
train['weekday'] = train['visit_date'].apply(lambda x:x.weekday())
train['weekofyear'] = train['visit_date'].apply(lambda x:x.isocalendar()[1])
train['month'] = train['visit_date'].apply(lambda x:x.month)
train['quater'] = train['visit_date'].apply(lambda x:FiscalDate(year=x.year, month=x.month, day=x.day).quarter)
train['year'] = train['visit_date'].apply(lambda x:x.year)
train['hour'] = train['visit_time'].apply(lambda x:x.hour)
train['minute'] = train['visit_time'].apply(lambda x:x.minute)
#train['second'] = train['visit_time'].apply(lambda x:x.second)
train.drop(columns=['visit_date', 'visit_time'], axis=1, inplace=True)

test['day'] = test['visit_date'].apply(lambda x:x.day)
test['weekday'] = test['visit_date'].apply(lambda x:x.weekday())
test['weekofyear'] = test['visit_date'].apply(lambda x:x.isocalendar()[1])
test['month'] = test['visit_date'].apply(lambda x:x.month)
test['quater'] = test['visit_date'].apply(lambda x:FiscalDate(year=x.year, month=x.month, day=x.day).quarter)
test['year'] = test['visit_date'].apply(lambda x:x.year)
test['hour'] = test['visit_time'].apply(lambda x:x.hour)
test['minute'] = test['visit_time'].apply(lambda x:x.minute)
#test['second'] = test['visit_time'].apply(lambda x:x.second)
test.drop(columns=['visit_date', 'visit_time'], axis=1, inplace=True)

train['visit_number_min_per_user'] = train.groupby('fullVisitorId')['visitNumber'].transform('min')
train['visit_number_max_per_user'] = train.groupby('fullVisitorId')['visitNumber'].transform('max')
train['visit_numer_per_user'] = train[['visitNumber','visit_number_min_per_user', 'visit_number_max_per_user']].apply(get_visit_numer_per_visit, axis=1)
train.drop(columns=['visit_number_min_per_user', 'visit_number_max_per_user'], axis=1, inplace=True)

test['visit_number_min_per_user'] = test.groupby('fullVisitorId')['visitNumber'].transform('min')
test['visit_number_max_per_user'] = test.groupby('fullVisitorId')['visitNumber'].transform('max')
test['visit_number_per_user'] = test[['visitNumber','visit_number_min_per_user', 'visit_number_max_per_user']].apply(get_visit_numer_per_visit, axis=1)
test.drop(columns=['visit_number_min_per_user', 'visit_number_max_per_user'], axis=1, inplace=True)

# bounce - minmax scaler
# hits, pageview,users, newuser - log transform, minmax scaler
# 

#------------------------------------------------------------------------------------------------------------
#train.to_csv('./train_pp.csv', index=False)
#test.to_csv('./test_pp.csv', index=False)

#del total_users_per_country, total_visits_per_country, total_hits_per_country, total_pageviews_per_country, visited_users
#del train_index, test_index

# 'visitId', 'totals_hits', 'totals_pageviews'

#train = pd.get_dummies(train, columns=['traffic_slot', 'traffic_adNetworkType'], drop_first=False)
#train['traffic_slot'].fillna(value='NL', inplace=True)
#train['traffic_adNetworkType'].fillna(value='NL', inplace=True)

#test = pd.get_dummies(test, columns=['traffic_slot', 'traffic_adNetworkType'], drop_first=False)
#test['traffic_slot'].fillna(value='NL', inplace=True)
#test['traffic_adNetworkType'].fillna(value='NL', inplace=True)

#t = train[['fullVisitorId', 'visitNumber', 'visit_numer_per_user', 'totals_transactionRevenue']]
#t = t.sort_values(by=['fullVisitorId', 'visitNumber'])


train['visitNumber'] = train['visitNumber'].apply(np.log1p)
train['totals_hits'] = train['totals_hits'].apply(np.log1p)
train['totals_pageviews'] = train['totals_pageviews'].apply(np.log1p)
#train['totals_transactionRevenue'] = train['totals_transactionRevenue'].apply(np.log1p)

test['visitNumber'] = test['visitNumber'].apply(np.log1p)
test['totals_hits'] = test['totals_hits'].apply(np.log1p)
test['totals_pageviews'] = test['totals_pageviews'].apply(np.log1p)

cols_to_scale = ['visitNumber', 'totals_hits', 'totals_pageviews', 
#                 'traffic_adclick_page',
                 'day', 'year', 'hour', 'minute'
#                 'traffic_keyword_count', 'traffic_keyword_count_unique'
#                 'next_revisit_time','prev_revisit_time'
                 ]

#cols_to_scale += cols_to_log_transform

for col in all_cat_cols:
    train[col].fillna(value='NULL', inplace=True)
    test[col].fillna(value='NULL', inplace=True)

for col in cols_to_rank_encode:
    train[col].fillna(value='NULL', inplace=True)
    test[col].fillna(value='NULL', inplace=True)
 
new_cols = {}
for col in cols_to_rank_encode:
    new_cols[col+'_freq_ratio'] = np.zeros(train.shape[0],dtype=np.float32)
 

stat_dict = {}
for col in cols_to_rank_encode:
    stat_dict[col+'_ct'] = {}
    stat_dict[col+'_cr'] = {}
    
for col in cols_to_rank_encode:
    print('runnig col:', col)
    arr_index = 0
    for i in train.index:
        value = train.loc[i, col]
        
        # case: occuring first time
        flag = True
        if value not in stat_dict[col+'_ct']:
            new_cols[col+'_freq_ratio'][arr_index] = 0.0
            arr_index += 1
            flag = False
            
        # case: not occuring first time
        if flag:     
            new_cols[col+'_freq_ratio'][arr_index] = stat_dict[col+'_cr'][value] / stat_dict[col+'_ct'][value]
            arr_index += 1
             
        stat_dict[col+'_ct'][value]  = stat_dict[col+'_ct'].get(value, 0) + 1
        
        if value not in stat_dict[col+'_cr']:
            stat_dict[col+'_cr'][value] = 0
            
        if train.loc[i, 'totals_transactionRevenue'] > 0.0:
            stat_dict[col+'_cr'][value] += 1
        
for col in cols_to_rank_encode:
    train[col+'_freq_ratio'] = new_cols[col+'_freq_ratio']
        
train[cols_to_scale] = get_scaled_data(train[cols_to_scale])
test[cols_to_scale] = get_scaled_data(test[cols_to_scale], is_training=False)

cols_to_drop_again = ['fullVisitorId', 'sessionId', 'visitId',
                      'weekofyear'
#                      'geonet_country', 'traffic_keyword',
                        ]

train.drop(columns = cols_to_drop_again, axis=1, inplace=True)
test.drop(columns = cols_to_drop_again, axis=1, inplace=True)

###############################################################################
#tx = train
#ty = train['totals_transactionRevenue'].apply(lambda x:1 if x>0 else 0)
#
#cat_feat_indices = [i for i in range(len(train.columns)) if train.columns[i] in all_cat_cols]
#
#print('Original size:\n', ty.value_counts())
#
#print('running smote-nc ...')
#msmote = SMOTENC(sampling_strategy = 0.5, 
#                 categorical_features = cat_feat_indices,
#                 k_neighbors = 5,
#                 random_state=2018, n_jobs=6)
#x_res, y_res = msmote.fit_resample(tx,ty)
#print('Size after msmote:\n', pd.Series(y_res).value_counts())
#
#x_res = pd.DataFrame(x_res, columns= tx.columns)
#x_res = pd.get_dummies(x_res, columns=all_cat_cols, drop_first=True)

###############################################################################
train = pd.get_dummies(train, columns=all_cat_cols, drop_first=True)
test = pd.get_dummies(test, columns=all_cat_cols, drop_first=True)

cols_to_consider = list(train.columns)
cols_to_consider.remove('totals_transactionRevenue')

for col in cols_to_consider:
    if col not in test.columns:
        test[col] = 0.0
    
test = test[cols_to_consider]

train.to_csv('./train_pp.csv', index=False)
test.to_csv('./test_pp.csv', index=False)

train.drop(columns = cols_to_rank_encode, axis=1, inplace=True)

#train = pd.read_csv('./train_pp.csv')
#train = pd.io.parsers.read_csv('./train_pp.csv', iterator = True, chunksize = 1000)
#train = pd.concat(train, ignore_index=True)
#
##test = pd.read_csv('./test_pp.csv')
#test = pd.io.parsers.read_csv('./test_pp.csv', iterator = True, chunksize = 1000)
#test = pd.concat(test, ignore_index=True)

#to_exclude_encoding = ['weekday', 'month']
#
#for col in all_cat_cols:
#    if train[col].isnull().sum() > 0:
#        train[col].fillna(value='NULL', inplace=True)
#       
#for col in all_cat_cols:
#    if test[col].isnull().sum() > 0:
#        test[col].fillna(value='NULL', inplace=True)
#        
#for col in all_cat_cols:
#    if col not in to_exclude_encoding:
#        le = LabelEncoder()
#        train_vals = list(train[col].values)
#        test_vals = list(test[col].values)
#    
#        le.fit(train_vals + test_vals)
#        train[col] = le.transform(train_vals)
#        test[col] = le.transform(test_vals)
    
tx = train
ty = train['totals_transactionRevenue'].apply(lambda x:1 if x>0 else 0)
#
#cols_for_pca = list(train.columns)
#cols_for_pca.remove('totals_transactionRevenue') 
#
#tx = get_transformed_data(tx[cols_for_pca], n_components=30)
#tx = pd.concat([tx, train['totals_transactionRevenue']], axis=1)
#
#test = get_transformed_data(test, is_training=False)

print('running sampling ...')
#sampling = SMOTEENN(sampling_strategy=0.5, random_state=102)
#x_res, y_res = sampling.fit_resample(tx,ty)

#print('Original size:\n', ty.value_counts())

#print('running msmote ...')
#msmote = BorderlineSMOTE(sampling_strategy = 0.5, random_state=2018, n_jobs=6)
#x_res, y_res = msmote.fit_resample(tx,ty)
#print('Size after msmote:\n', pd.Series(y_res).value_counts())
#
##print('running smote ...')
##smote = SMOTE(sampling_strategy = 0.5, random_state=2018, n_jobs=6)
##x_res, y_res = smote.fit_resample(tx,ty)
##print('Size after smote:\n', pd.Series(y_res).value_counts())
#
#print('running enn ...')
#enn = RepeatedEditedNearestNeighbours(sampling_strategy = 'majority', random_state=2018,
#                                      max_iter = 5, n_jobs=6)
#x_res, y_res = enn.fit_resample(x_res, y_res) 
#print('Size after enn:\n', pd.Series(y_res).value_counts())
#
#print('running tls ...')
#tls = TomekLinks(sampling_strategy = 'majority', random_state=2018, n_jobs=6)
#x_res, y_res = tls.fit_resample(x_res, y_res)
#print('Size after tlk:\n', pd.Series(y_res).value_counts())
#
#x_res = pd.DataFrame(x_res,columns=tx.columns)
#y_res = pd.Series(y_res,name=ty.name)

#x_res = pd.DataFrame(x_res,columns=train.columns)
#y_res = pd.Series(y_res,name=ty.name)

#x_res.to_csv('../results/x_res.csv', index=False) 
#y_res.to_csv('../results/y_res.csv', index=False)

#x_res = pd.read_csv('../results/x_res.csv')
#y_res = pd.read_csv('../results/y_res.csv')

#x = x_res.drop('totals_transactionRevenue', axis=1)
#y = x_res['totals_transactionRevenue']

x = train.drop('totals_transactionRevenue', axis=1)
y = train['totals_transactionRevenue']

#x = get_transformed_data(x, n_components=30)
#test = get_transformed_data(test, is_training=False)

test_full_visitor_id = pd.concat([test_full_visitor_id, train_full_visitor_id[test_extra_index]], axis=0, ignore_index=True)
test = pd.concat([test, x.loc[test_extra_index, :]], axis=0, ignore_index=True)

#xgb_regressor = xgb.XGBRegressor(max_depth = 8,
#                                   n_estimators = 1000,
#                                   learning_rate = 0.01,
#                                   subsample = 0.8,
#                                   colsample_bytree = 0.7,
#                                   
#                                   reg_alpha = 0.2,
#                                   reg_lambda = 0.2 ,
#                                   
##                                   scale_pos_weight = ratio,
#                                   booster = 'gbtree',
##                                   rate_drop = 0.3,
#                                   
##                                   objective = 'binary:logistic',
##                                   eval_metric = 'rmse',
#                                   n_jobs=-1,
#                                   tree_method = 'hist',
#                                   random_state = 2018,
##                                   max_bin = 64
#                                   )
#
#k = 10
##skf = StratifiedKFold(n_splits=k, shuffle=True)
##tscv = TimeSeriesSplit(n_splits=k)
#kf = KFold(n_splits=k)
#
##prediction = np.zeros(test.shape[0])
#
#for fold_i, (train_index, test_index) in enumerate(kf.split(x)):
#    print('fold:', fold_i)
#    x_train, x_test = x.loc[train_index, :], x.loc[test_index, :]
#    y_train, y_test = y[train_index], y[test_index]
#    
#    xgb_regressor.fit(x_train, y_train,
#                       eval_set=[(x_train, y_train), (x_test, y_test)],
#                       eval_metric = 'rmse',
#                       early_stopping_rounds = 100
##                       verbose = True
#                       )
#    
#
#xgb_regressor.save_model('../models/xgb_regressor_1.model')

#xgb_regressor = xgb.Booster({'nthread': 4})
#xgb_regressor.load_model('../models/xgb_regressor_1.model')  


lgb_regressor = lgb.LGBMRegressor(objective = 'regression', 
                                  metric = 'rmse', 
                                  max_depth = 12, 
                                  min_child_samples =  20, 
                                  
                                  reg_alpha =  0.3, 
                                  reg_lambda =  0.3,
                                  num_leaves =  257, 
                                  
                                  learning_rate =  0.01,
                                  n_estimators = 1000, 
                                  
                                  subsample = 0.8, 
                                  colsample_bytree =  0.7, 
                                  subsample_freq =  5,
                                  
                                  device = 'gpu',
                                  gpu_platform_id = 1,
                                  gpu_device_id = 0 
                                  )

k = 8
#skf = StratifiedKFold(n_splits=k, shuffle=True)
#yt = y.apply(lambda x:1 if x>0 else 0)

tskf = TimeSeriesSplit(n_splits=k)
for fold_i, (train_index, test_index) in enumerate(tskf.split(x)):
    print('fold:', fold_i)
    x_train, x_test = x.loc[train_index, :], x.loc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

    lgb_regressor.fit(x_train, y_train, 
              eval_set=[(x_train, y_train), (x_test, y_test)], eval_metric='rmse',
              verbose=500, early_stopping_rounds=100)
         
cols = list(train.columns)
cols.remove('totals_transactionRevenue')

def predict(model, data):
    for col in cols_to_rank_encode:  
        value = data[col]
        if value not in stat_dict[col+'_ct']:
            data[col+'_freq_ratio'] = 0.0
        else:
            data[col+'_freq_ratio'] = stat_dict[col+'_cr'][value] / stat_dict[col+'_ct'][value]            
    
    sample = np.array([data[col] for col in cols])
    
    data['totals_transactionRevenue'] = model.predict(sample.reshape(1,-1))
    
    for col in cols_to_rank_encode:
        value = data[col]
        
        stat_dict[col+'_ct'][value]  = stat_dict[col+'_ct'].get(value, 0) + 1
        
        if value not in stat_dict[col+'_cr']:
            stat_dict[col+'_cr'][value] = 0
            
        if data['totals_transactionRevenue'] > 0.0:
            stat_dict[col+'_cr'][value] += 1
            
    return data['totals_transactionRevenue'][0]
        
    
y_pred_reg = [predict(lgb_regressor, test.loc[i,:].to_dict()) for i in test.index]
y_pred_reg = pd.Series(y_pred_reg).apply(lambda x:0.0 if x<0 else x)
#y_pred_reg = pd.Series(lgb_regressor.predict(test))

submission = pd.concat([test_full_visitor_id, y_pred_reg], axis=1)
submission.columns = ['fullVisitorId', 'PredictedLogRevenue']

submission['PredictedLogRevenue'].fillna(value=0.0, inplace=True)
submission = submission.groupby(by='fullVisitorId').agg({'PredictedLogRevenue':sum})
submission['PredictedLogRevenue'].fillna(value=0.0, inplace=True)


submission.reset_index(drop=False, inplace=True)

sample_submission.drop('PredictedLogRevenue', axis=1, inplace=True)
submission = pd.merge(sample_submission, submission, how='left', on='fullVisitorId')

submission['PredictedLogRevenue'].fillna(value=0.0, inplace=True)
submission['PredictedLogRevenue'] = submission['PredictedLogRevenue'].apply(lambda x:x if x>0 else 0.0)

submission['PredictedLogRevenue'] = submission['PredictedLogRevenue'].apply(np.log1p)

submission.to_csv('./submission.csv', index=False)
