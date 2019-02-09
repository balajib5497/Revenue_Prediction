import numpy as np
import pandas as pd
import datetime as dt
from fiscalyear import FiscalDate

from sklearn.model_selection import KFold, TimeSeriesSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

from sklearn.naive_bayes import MultinomialNB

from util_naive_bayes import *

from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize, dump, load
from skopt.plots import plot_convergence

print('reading data & json blobs cleaning ...')

#train = pd.read_csv('../dataset/train.csv')
#train = clean_json_blobs(train, is_train=True)
#test = pd.read_csv('../dataset/test.csv')
#test = clean_json_blobs(test)

#train.to_csv('./train_p.csv', index=False)
#test.to_csv('./test_p.csv', index=False)

train = pd.read_csv('./train_p.csv')
test = pd.read_csv('./test_p.csv')

print('starts preprocessing ...')

#train = train[~train.duplicated(subset=['sessionId'])]

# google - google, goog
# youtube - youtube, you, tube, 
# android

train['fullVisitorId'] = train['sessionId'].apply(lambda x:x.split('_')[0])
train_full_visitor_id = train['fullVisitorId']

test['fullVisitorId'] = test['sessionId'].apply(lambda x:x.split('_')[0])
test_full_visitor_id = test['fullVisitorId']

sample_submission = pd.read_csv('../dataset/sample_submission.csv')
sample_submission_ids = sample_submission['fullVisitorId'].to_dict()
sample_submission_ids = {int(value):key for key, value in sample_submission_ids.items()}
test_extra_index = [i for i in train_full_visitor_id.index if int(train_full_visitor_id[i]) in sample_submission_ids]

train['traffic_keyword'] = train['traffic_keyword'].apply(token_normalization)
train['traffic_keyword_count'] = train.groupby('traffic_keyword')['sessionId'].transform('nunique')
train['traffic_keyword_count_unique'] = train.groupby('traffic_keyword')['fullVisitorId'].transform('nunique')

test['traffic_keyword'] = test['traffic_keyword'].apply(token_normalization)
test['traffic_keyword_count'] = test.groupby('traffic_keyword')['sessionId'].transform('nunique')
test['traffic_keyword_count_unique'] = test.groupby('traffic_keyword')['fullVisitorId'].transform('nunique')

traffic_keywords = pd.concat([train['traffic_keyword'], test['traffic_keyword']], axis=0,ignore_index=True)
traffic_keywords = traffic_keywords.value_counts().reset_index(drop=False)
traffic_keywords.columns  = ['traffic_keyword', 'count']

traffic_adContents = pd.concat([train['traffic_adContent'], test['traffic_adContent']], axis=0,ignore_index=True)
traffic_adContents = traffic_adContents.value_counts().reset_index(drop=False)
traffic_adContents.columns  = ['traffic_adContent', 'count']

chars_to_remove = set(['(', ')', '+', ''])
chars_to_replace = set(['.', '-', '==', '/', '::', '*', ','])

# yet to handle
# - link, other langaue, remove numbers

for col in (cols_to_fill_zero + ['totals_transactionRevenue']):
    train[col].fillna(value=0, inplace=True)
    train[col] = train[col].astype(int)

# train['traffic_adclickinfo_isnull'] = train['traffic_gclId'].apply(lambda x: 1.0 if type(x) != str and math.isnan(x) else 0.0)
train.drop(columns=cols_to_drop, axis=1, inplace=True)  

train['totals_hits'] = train['totals_hits'].astype(float)
train['visitStartTime'] = train['visitStartTime'].apply(lambda x:dt.datetime.fromtimestamp(x))
train['visit_date'] = pd.to_datetime(train['visitStartTime'].apply(lambda x:x.date()))
train['visit_time'] = train['visitStartTime'].apply(lambda x:x.time())
train.drop('visitStartTime', axis=1, inplace=True)

#train['date'] = train['date'].apply(to_date)
#train['date_diff'] = (train['visit_date'] - train['date']).apply(lambda x: int(str(x)[0]))
train['visit_date'] = train['visit_date'].apply(lambda x:x.date())
date_col = train['date'].copy()
train.drop('date', axis=1, inplace=True)

#train['device_browser'] = train['device_browser'].apply(eliminate_low_freq_browsers)
train['traffic_isTrueDirect'] = train['traffic_isTrueDirect'].apply(value_or_null, args=(True, ))
train['traffic_adclick_isVideoAd'] = train['traffic_adclick_isVideoAd'].apply(value_or_null, args=(False, ))

for col in cols_to_fill_zero:
    test[col].fillna(value=0, inplace=True)
    test[col] = test[col].astype(int)
    

# test['traffic_adclickinfo_isnull'] = test['traffic_gclId'].apply(lambda x: 1.0 if type(x) != str and math.isnan(x) else 0.0)
test.drop(columns=cols_to_drop, axis=1, inplace=True)  

test['totals_hits'] = test['totals_hits'].astype(float)
test['visitStartTime'] = test['visitStartTime'].apply(lambda x:dt.datetime.fromtimestamp(x))
test['visit_date'] = pd.to_datetime(test['visitStartTime'].apply(lambda x:x.date()))
test['visit_time'] = test['visitStartTime'].apply(lambda x:x.time())
test.drop('visitStartTime', axis=1, inplace=True)

#test['date'] = test['date'].apply(to_date)
#test['date_diff'] = (test['visit_date'] - test['date']).apply(lambda x: int(str(x)[0]))
test['visit_date'] = test['visit_date'].apply(lambda x:x.date())
test.drop('date', axis=1, inplace=True)

#test['device_browser'] = test['device_browser'].apply(eliminate_low_freq_browsers)
test['traffic_isTrueDirect'] = test['traffic_isTrueDirect'].apply(value_or_null, args=(True, ))
test['traffic_adclick_isVideoAd'] = test['traffic_adclick_isVideoAd'].apply(value_or_null, args=(False, ))

train['day'] = train['visit_date'].apply(lambda x:x.day)
train['weekday'] = train['visit_date'].apply(lambda x:x.weekday())
#train['weekofyear'] = train['visit_date'].apply(lambda x:x.isocalendar()[1])
train['month'] = train['visit_date'].apply(lambda x:x.month)
train['quater'] = train['visit_date'].apply(lambda x:FiscalDate(year=x.year, month=x.month, day=x.day).quarter)
train['year'] = train['visit_date'].apply(lambda x:x.year)
train['hour'] = train['visit_time'].apply(lambda x:x.hour)
#train['minute'] = train['visit_time'].apply(lambda x:x.minute)
#train['second'] = train['visit_time'].apply(lambda x:x.second)
train.drop(columns=['visit_date', 'visit_time'], axis=1, inplace=True)

test['day'] = test['visit_date'].apply(lambda x:x.day)
test['weekday'] = test['visit_date'].apply(lambda x:x.weekday())
#test['weekofyear'] = test['visit_date'].apply(lambda x:x.isocalendar()[1])
test['month'] = test['visit_date'].apply(lambda x:x.month)
test['quater'] = test['visit_date'].apply(lambda x:FiscalDate(year=x.year, month=x.month, day=x.day).quarter)
test['year'] = test['visit_date'].apply(lambda x:x.year)
test['hour'] = test['visit_time'].apply(lambda x:x.hour)
#test['minute'] = test['visit_time'].apply(lambda x:x.minute)
#test['second'] = test['visit_time'].apply(lambda x:x.second)
test.drop(columns=['visit_date', 'visit_time'], axis=1, inplace=True)

#train['visit_number_min_per_user'] = train.groupby('fullVisitorId')['visitNumber'].transform('min')
#train['visit_number_max_per_user'] = train.groupby('fullVisitorId')['visitNumber'].transform('max')
#train['visit_number_per_user'] = train[['visitNumber','visit_number_min_per_user', 'visit_number_max_per_user']].apply(get_visit_numer_per_visit, axis=1)
#train.drop(columns=['visit_number_min_per_user'], axis=1, inplace=True)
#
#test['visit_number_min_per_user'] = test.groupby('fullVisitorId')['visitNumber'].transform('min')
#test['visit_number_max_per_user'] = test.groupby('fullVisitorId')['visitNumber'].transform('max')
#test['visit_number_per_user'] = test[['visitNumber','visit_number_min_per_user', 'visit_number_max_per_user']].apply(get_visit_numer_per_visit, axis=1)
#test.drop(columns=['visit_number_min_per_user'], axis=1, inplace=True)

#sort_by_cols = ['year', 'month', 'day', 'hour', 'minute', 'second']
#total_users_per_country = {}
#total_visits_per_country = {}
#total_hits_per_country = {}
#total_pageviews_per_country = {}
#visited_users = {}
#
#def generate_values_from_country(data):
#    for i, country, user, total_hits, totals_pageviews in zip(range(data.shape[0]), data['geonet_country'],data['fullVisitorId'], data['totals_hits'], data['totals_pageviews']) :
#        if country not in visited_users:
#            visited_users[country] = {user:True}
#            total_users_per_country[country] = 1
#        else:
#            if user not in visited_users[country]:
#                visited_users[country][user] = True
#                total_users_per_country[country] += 1 
#        data_total_users_per_country[i] = total_users_per_country[country]
#        
#        data_total_visits_per_country[i] = total_visits_per_country[country] = total_visits_per_country.get(country, 0) + 1 
#        
#        data_total_hits_per_country[i] = total_hits_per_country[country] = total_hits_per_country.get(country, 0) + total_hits
#        
#        data_total_pageviews_per_country[i] = total_pageviews_per_country[country] = total_pageviews_per_country.get(country, 0) + totals_pageviews
#    
#train_index = list(train.index)
#train.sort_values(by=sort_by_cols, inplace=True)
#
#data_total_users_per_country = np.zeros(train.shape[0])
#data_total_visits_per_country = np.zeros(train.shape[0])
#data_total_hits_per_country = np.zeros(train.shape[0])
#data_total_pageviews_per_country = np.zeros(train.shape[0])
#
#generate_values_from_country(train)
#
#train['total_users_per_country'] = data_total_users_per_country
#train['total_visits_per_country'] = data_total_visits_per_country
#train['total_hits_per_country'] = data_total_hits_per_country
#train['total_pageviews_per_country'] = data_total_pageviews_per_country
#
#train['avg_hits_per_country'] = train['total_hits_per_country'] / train['total_visits_per_country']    
#train['avg_pageviews_per_country'] = train['total_pageviews_per_country'] / train['total_visits_per_country']    
#train = train.loc[train_index, :]
#
#test_index = list(test.index)
#test.sort_values(by=sort_by_cols, inplace=True)
#
#data_total_users_per_country = np.zeros(test.shape[0])
#data_total_visits_per_country = np.zeros(test.shape[0])
#data_total_hits_per_country = np.zeros(test.shape[0])
#data_total_pageviews_per_country = np.zeros(test.shape[0])
#
#generate_values_from_country(test)   
#
#test['total_users_per_country'] = data_total_users_per_country
#test['total_visits_per_country'] = data_total_visits_per_country
#test['total_hits_per_country'] = data_total_hits_per_country
#test['total_pageviews_per_country'] = data_total_pageviews_per_country
#test['avg_hits_per_country'] = test['total_hits_per_country'] / test['total_visits_per_country']    
#test['avg_pageviews_per_country'] = test['total_pageviews_per_country'] / test['total_visits_per_country']     
#test = test.loc[test_index, :]

#
#cols_to_log_transform = [col for col in train.columns if col.startswith('users_per') or col.startswith('new_users_per') or col.startswith('pageviews_per') or col.startswith('hits_per')]
#
#for col in cols_to_log_transform:
#    train[col] = train[col].apply(np.log1p)
#
#for col in cols_to_log_transform:
#    test[col] = test[col].apply(np.log1p)
#train.to_csv('./train_pp.csv', index=False)
#test.to_csv('./test_pp.csv', index=False)

#del total_users_per_country, total_visits_per_country, total_hits_per_country, total_pageviews_per_country, visited_users
#del train_index, test_index

# 'visitId', 'totals_hits', 'totals_pageviews'



#train['month_unique_user_count'] = train.groupby('month')['fullVisitorId'].transform('nunique')
#train['day_unique_user_count'] = train.groupby('day')['fullVisitorId'].transform('nunique')
#train['weekday_unique_user_count'] = train.groupby('weekday')['fullVisitorId'].transform('nunique')
#
#train['mean_hits_per_day'] = train.groupby(['day'])['totals_hits'].transform('mean')
#train['sum_hits_per_day'] = train.groupby(['day'])['totals_hits'].transform('sum')
#
#train['user_pageviews_sum'] = train.groupby('fullVisitorId')['totals_pageviews'].transform('sum')
#train['user_hits_sum'] = train.groupby('fullVisitorId')['totals_hits'].transform('sum')
#
#train['user_pageviews_count'] = train.groupby('fullVisitorId')['totals.pageviews'].transform('count')
#train['user_hits_count'] = train.groupby('fullVisitorId')['totals_hits'].transform('count')
#
#train['user_pageviews_sum_to_mean'] = train['user_pageviews_sum'] / train['user_pageviews_sum'].mean()
#train['user_hits_sum_to_mean'] = train['user_hits_sum'] / train['user_hits_sum'].mean()
#
#test['mean_hits_per_day'] = test.groupby(['day'])['totals.hits'].transform('mean')
#test['sum_hits_per_day'] = test.groupby(['day'])['totals.hits'].transform('sum')


#train = pd.get_dummies(train, columns=['traffic_slot', 'traffic_adNetworkType'], drop_first=False)
#train['traffic_slot'].fillna(value='NL', inplace=True)
#train['traffic_adNetworkType'].fillna(value='NL', inplace=True)

#test = pd.get_dummies(test, columns=['traffic_slot', 'traffic_adNetworkType'], drop_first=False)
#test['traffic_slot'].fillna(value='NL', inplace=True)
#test['traffic_adNetworkType'].fillna(value='NL', inplace=True)

#t = train[['fullVisitorId', 'visitNumber', 'visit_numer_per_user', 'totals_transactionRevenue']]
#t = t.sort_values(by=['fullVisitorId', 'visitNumber'])


#train['visitNumber'] = train['visitNumber'].apply(np.log1p)
#train['totals_hits'] = train['totals_hits'].apply(np.log1p)
#train['totals_pageviews'] = train['totals_pageviews'].apply(np.log1p)
train['totals_transactionRevenue'] = train['totals_transactionRevenue'].apply(np.log1p)
totals_transactionRevenue = train['totals_transactionRevenue'].values
train['transaction_status'] = train['totals_transactionRevenue'].apply(lambda x:1 if x>0 else 0)

#test['visitNumber'] = test['visitNumber'].apply(np.log1p)
#test['totals_hits'] = test['totals_hits'].apply(np.log1p)
#test['totals_pageviews'] = test['totals_pageviews'].apply(np.log1p)


#cols_to_scale = ['visitNumber', 'totals_hits', 'totals_pageviews', 'traffic_adclick_page',
#                 'day', 'month', 'year', 'hour', 'minute']
#
#cols_to_scale += cols_to_log_transform
#
#train[cols_to_scale] = get_scaled_data(train[cols_to_scale])
#test[cols_to_scale] = get_scaled_data(test[cols_to_scale], is_training=False)

cols_to_drop_again = ['fullVisitorId', 'sessionId', 'visitId',
#                      'weekofyear',
#                      'geonet_country', 'geonet_networkDomain', 'traffic_source',
                      'totals_transactionRevenue'
                        ]

train.drop(columns = cols_to_drop_again, axis=1, inplace=True)
cols_to_drop_again.remove('totals_transactionRevenue')

test.drop(columns = cols_to_drop_again, axis=1, inplace=True)

#train = pd.get_dummies(train, columns=all_cat_cols, drop_first=True)
#test = pd.get_dummies(test, columns=all_cat_cols, drop_first=True)

cols_to_consider = list(train.columns)
cols_to_consider.remove('transaction_status')

for col in cols_to_consider:
    if col not in test.columns:
        test[col] = 0.0
    
test = test[cols_to_consider]

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
for col in all_cat_cols:
    le = LabelEncoder()
    train_vals = list(train[col].values)
    test_vals = list(test[col].values)

    le.fit(train_vals + test_vals)
    train[col] = le.transform(train_vals)
    test[col] = le.transform(test_vals)
    
x = train.drop('transaction_status', axis=1)
y = train['transaction_status']

#sampling = SMOTEENN(ratio=0.5, random_state=102)
#x_res, y_res = sampling.fit_resample(x,y)

#x_res = pd.DataFrame(x_res,columns=x.columns)
#y_res = pd.Series(y_res,name=y.name)

test_full_visitor_id = pd.concat([test_full_visitor_id, train_full_visitor_id[test_extra_index]], axis=0, ignore_index=True)
test = pd.concat([test, x.loc[test_extra_index, :]], axis=0, ignore_index=True)
###############################################################################
#space  = [Integer(8, 30, name='max_depth'),
#          Integer(20, 150, name='num_leaves'),
#          Integer(20, 100, name='min_data_in_leaf'),
#          Real(0.5, 0.9, name='feature_fraction'),
#          Real(0.5, 0.9, name='bagging_fraction'),
#          Real(0.1, 0.5, name='lambda_l1'),
#          Real(0.1, 0.5, name='lambda_l2'),
##         Real(0.1, 0.5, name='drop_rate'),
#         Real(0.001, 0.1,  name='learning_rate'),
##         Categorical(['gbdt', 'rf', 'dart'], name='boosting_type'),
#          Integer(1, 10, name='bagging_freq'),
##          Integer(5,20, name='nfold'),
##          Integer(50,1000, name='n_estimators')
#         ]
##
#def objective(values):
#    global count
#    params = {'max_depth': values[0], 
#              'num_leaves': values[1], 
#              'min_data_in_leaf': values[2], 
#              'feature_fraction': values[3], 
#              'bagging_fraction': values[4],
#              'lambda_l1': values[5],
#              'lambda_l2': values[6],
##              'drop_rate': values[6],
#              'learning_rate': values[7],
#              'num_boost_round':1000,
##              'boosting_type': 'gbdt',
#              'bagging_freq':values[8],
#              
#              'objective' :'regression', 
#              'metric' :'rmse',
#              'seed':35,
##              'early_stopping_round':100,
#              'num_threads':-1,
#              'device':'gpu',
#              'gpu_platform_id':1,
#              'gpu_device_id':0,
#              'gpu_use_dp':True,
##              'max_bin':63
#             }
#    
#    print('\nNext set of params.....',params)
#    
##    early_stopping_rounds = 100
#    nfold = 7
##    num_boost_round = 100
#    
#    train_data = lgb.Dataset(data = x, 
#                          label = y,
#                          categorical_feature = all_cat_cols,
#                          )
#    tscv = TimeSeriesSplit(n_splits=nfold)
#    
#    print('running cv')
#    cv_results = lgb.cv(params,
#                             categorical_feature = all_cat_cols,
#                             train_set =  train_data,
##                             num_boost_round = num_boost_round, 
##                             early_stopping_rounds = early_stopping_rounds,
#                             folds = tscv,
#                             nfold = nfold,
#                             stratified = False,
#                             shuffle = False)
#    
##    param_file = '../results/params_' + str(count) + '.txt'
##    cv_results_file = '../results/cv_result_' + str(count) + '.txt'
#    print(cv_results)
##    with open(param_file, 'w') as fp:
##        fp.write(json.dumps(params,encoding='UTF-8',default=str))
##        
##    with open(cv_results_file, 'w') as fp:
##        fp.write(json.dumps(cv_results, encoding='UTF-8',default=str))
#    
#    print('count: ', count)
#    count += 1
#    print('best n_estimators:', len(cv_results['rmse-mean']))
#    print('best cv score:', cv_results['rmse-mean'][-1])
#    return  cv_results['rmse-mean'][-1]
#
#count = 0
#
#print('runing gp_minimize ...')
#res_gp = gp_minimize(objective, space, n_calls=200, random_state=13)
#dump(res_gp, '../results/res_gp.pkl')

#print('Best score=%.4f' % res_gp.fun)
##print("""Best parameters:
##    - max_depth=%d
##    - min_data_in_leaf=%d
##    - feature_fraction=%.4f
##    - bagging_fraction=%.4f
##    - lambda_l1=%.4f
##    - lambda_l2=%.4f
##    - learning_rate=%.4f
##    - boosting_type=%s
##    - bagging_freq=%d
##    - nfold=%d
##    - n_estimators""" % (res_gp.x[0], res_gp.x[1], res_gp.x[2], res_gp.x[3], res_gp.x[4],res_gp.x[5], 
##    res_gp.x[6],res_gp.x[7], res_gp.x[8], res_gp.x[9], res_gp.x[10]))
#
#plot_convergence(res_gp)
###############################################################################

model = MultinomialNB()

k = 10
skf  = StratifiedKFold(n_splits=k)

#prediction = np.zeros(test.shape[0])

for fold_i, (train_index, test_index) in enumerate(skf.split(x,y)):
    print('fold:', fold_i)
    x_train, x_test = x.loc[train_index, :], x.loc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)  
    print('classification_report:\n', classification_report(y_test, y_pred))
    print('confusion_matrix:\n', confusion_matrix(y_test, y_pred))
    
min_trans = train[train['totals_transactionRevenue'] > 0]['totals_transactionRevenue'].min()

prediction = model.predict(test, num_iteration = model.best_iteration_)
prediction = pd.Series(prediction)
#prediction = prediction.apply(lambda x:x if x >= min_trans else 0.0)

submission = pd.concat([test_full_visitor_id, prediction], axis=1)
submission.columns = ['fullVisitorId', 'PredictedLogRevenue']
submission = submission.groupby(by='fullVisitorId').agg({'PredictedLogRevenue':sum})
submission.reset_index(drop=False, inplace=True)

sample_submission.drop('PredictedLogRevenue', axis=1, inplace=True)
submission = pd.merge(sample_submission, submission, how='left', on='fullVisitorId')

submission['PredictedLogRevenue'].fillna(value=0.0, inplace=True)

submission.to_csv('./submission.csv', index=False)
