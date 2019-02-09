import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA

import json
import datetime as dt
import math

from nltk.corpus import stopwords
from string import punctuation

scaler = None

t_scaler, pca = None, None

min_date = dt.date(year=2016, month=8, day=1)

common_word = set(['store', ' stores', 'storre', 'shop', 'shops', 'buy', 'item', 'items','co', 'com',
                              'http', 'https', 'www', 'httpwww', 'httpswww', 'sale', 'sales', 'price', 'online'])
cols_to_drop = ['socialEngagementType', 
                'traffic_adclick_criteriaParameters', 
                'totals_visits', 'traffic_medium',
                
#                'geonet_networkDomain', 
#                'geonet_city', 
#                'geonet_metro', 
#                'geonet_region', 
                
#                'traffic_adContent', 
#                'traffic_campaign', 
##                'traffic_keyword', 
#                'traffic_referralPath', 
#                'traffic_source',
                
##                'traffic_adclick_adNetworkType',
                'traffic_adclick_gclId',
##                'traffic_adclick_page',
##                'traffic_adclick_slot',
                ]
  
cols_to_fill_zero = ['totals_bounces', 'totals_newVisits', 'totals_pageviews', 
                     'totals_totalTransactionRevenue', 'totals_transactionRevenue', 'totals_transactions'
                    'traffic_adclick_page']

keep_browsers = ['chrome', 'safari', 'firefox', 'internet explorer', 'edge', 'android webview', 'safari (in-app)', 'opera mini',
                 'opera', 'uc browser', 'yabrowser', 'Coc Coc', 'amazon silk', 'android browser', 'mozilla compatible agent',
                 'mrchrome', 'maxthon ', 'blackberry', 'nintendo browser']

#cat_cols = ['channelGrouping', 'device_browser',1100 'device_deviceCategory',
#            'device_operatingSystem', 'geonet_continent', 'geonet_country',
#            'geonet_subContinent']


cat_cols_to_model = ['channelGrouping', 'device_browser', 'device_deviceCategory', 'device_operatingSystem',
                     'geonet_continent', 'geonet_subContinent',
#                     'traffic_slot', 'traffic_adNetworkType', 
                     'weekday', 'month','quater'
#                   'weekofyear'
                     ]

all_cat_cols = ['channelGrouping',  
                
                'device_browser', 
                'device_deviceCategory', 
                'device_operatingSystem',
##                'geonet_networkDomain', 
##                'geonet_city', 
##                'geonet_metro', 
##                'geonet_region', 
                'geonet_continent', 
                'geonet_subContinent', 
##                'geonet_country',
                
                'traffic_adContent', 
                'traffic_campaign', 
##                'traffic_keyword', 
##                'traffic_referralPath', 
##                'traffic_source',
                
                'traffic_adclick_adNetworkType',
##                'traffic_adclick_gclId',
                'traffic_adclick_page',
                'traffic_adclick_slot',
                
                'weekday', 'month', 'quater'
                ]

cols_to_rank_encode = [
#                'device_browser',
                
                'geonet_networkDomain', 
                'geonet_city', 
                'geonet_metro', 
                'geonet_region', 
                'geonet_country',

#                'traffic_adContent', 
#                'traffic_campaign', 
                'traffic_keyword', 
                'traffic_referralPath', 
                'traffic_source',
                
#                'traffic_adclick_gclId'
                ]

def clean_json_blobs(data, is_train=False):
    # device 
    device = [json.loads(data.loc[i, 'device']) for i in data.index]
#    device = data['device']
    data.drop('device', axis=1, inplace=True)
    device = pd.DataFrame.from_records(device)
    device = device[['browser', 'deviceCategory', 'isMobile', 'operatingSystem']]
    device.columns = ['device_' + col for col in device.columns]
  
    # geo_network
    geo_network = [json.loads(data.loc[i, 'geoNetwork']) for i in data.index]
#    geo_network = data['geoNetwork']
    data.drop('geoNetwork', axis=1, inplace=True)
    geo_network = pd.DataFrame.from_records(geo_network)
    geo_network = geo_network[['city', 'continent', 'country', 'metro', 'networkDomain', 'region', 'subContinent']]
    geo_network.columns = ['geonet_' + col for col in geo_network.columns]
  
    # traffic_source
    traffic_source = [json.loads(data.loc[i, 'trafficSource']) for i in data.index]
#    traffic_source = data['trafficSource']
    data.drop('trafficSource', axis=1, inplace=True)
    traffic_source = pd.DataFrame.from_records(traffic_source)
    
    adwordsClickInfo = pd.DataFrame.from_records(traffic_source['adwordsClickInfo'])
    traffic_source = traffic_source[['isTrueDirect', 'adContent', 'campaign', 'keyword', 'medium', 'referralPath', 'source']]
    adwordsClickInfo = adwordsClickInfo[['adNetworkType', 'criteriaParameters', 'gclId', 'isVideoAd', 'page', 'slot']]
    adwordsClickInfo.columns = ['adclick_' + col for col in adwordsClickInfo.columns]
    
    traffic_source = pd.concat([traffic_source, adwordsClickInfo], axis=1)
    del adwordsClickInfo
    
    traffic_source.columns = ['traffic_' + col for col in traffic_source.columns]
    
    # totals 
    totals = [json.loads(data.loc[i, 'totals']) for i in data.index]
#    totals = data['totals']
    data.drop('totals', axis=1, inplace=True)
    totals = pd.DataFrame.from_records(totals)
    if is_train:
        totals = totals[['bounces', 'newVisits', 'visits', 'hits', 'pageviews', 'transactionRevenue']]
    else:
        totals = totals[['bounces', 'newVisits', 'visits', 'hits', 'pageviews']]
    
    totals.columns = ['totals_' + col for col in totals.columns]
  
    data = pd.concat([data, device, geo_network, traffic_source, totals], axis=1)
    return data

def to_date(date):
    date = str(date)
    return dt.datetime(int(date[:4]), int(date[4:6]), int(date[6:]))

def eliminate_low_freq_browsers(x):
    if x not in keep_browsers:
        return 'Others'
    else:
        return x
  
def value_or_null(x, value):
    if x == value:
        return 1.0
    return 0.0

def get_scaled_data(data, is_training = True):
    global scaler 
    if is_training:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = scaler.transform(data)
        
    scaled_data = pd.DataFrame(scaled_data, columns = data.columns)
    return scaled_data

def get_visit_numer_per_visit(values):
    if values[1] == values[2]:
        return 1.0
    if values[0] == values[1]:
        return 0.0
    return (values[0] - values[1]) / (values[2] - values[1])

def will_buy_or_not(values):
    if values[0] == 0:
        return 0.0
    return values[1]

def count_new_users(values):
#    print(values)
    return 1
#    count = 0
#    for i in range(len(values[0])):
#        if values[0,i] == 1:
#            count += values[1,i]
#    return count
    
    
def token_normalization(text):
    if type(text) != str:
        return text
    text = text.lower()
    text = text.replace('.', ' ')
    text = text.strip()
#    tokenizer = TreebankWordTokenizer()
#    tokens = tokenizer.tokenize(text)
#    normalizer = WordNetLemmatizer()
    
    stopwds = stopwords.words('English')
#    text = ''.join([(c if c not in punctuation else ' ') for c in text])
    text = ''.join([c for c in text if c not in punctuation])
    tokens = text.split()
    tokens = [token for token in tokens if token not in stopwds]
    tokens = [token for token in tokens if token not in common_word]
#    tokens = [normalizer.lemmatize(token) for token in tokens]
    text = ' '.join(tokens)
    text = text.strip()
    if text.isspace():
        return 'NNN'
    return text


def get_transformed_data(x, n_components = None,is_training = True):
    global t_scaler, pca
    if is_training:
        t_scaler = MinMaxScaler()
        scaled_data = t_scaler.fit_transform(x)
        
        pca = PCA(n_components = n_components)
        res = pca.fit_transform(scaled_data)
    else:
        scaled_data = t_scaler.transform(x)
        res = pca.transform(scaled_data)
     
    res = pd.DataFrame(res)
    return res

def add_time_period_of_same_ID(df): 
    df['dt'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df.sort_values(['fullVisitorId', 'dt'], ascending=True, inplace=True)
    df['next_revisit_time'] = (df['dt'] - df[['fullVisitorId', 'dt']].groupby('fullVisitorId')['dt'].shift(1))
    df['next_revisit_time'] = df['next_revisit_time'].astype(np.int64) // 1e9 // 60 // 60
    
    df['prev_revisit_time'] = (df['dt'] - df[['fullVisitorId', 'dt']].groupby('fullVisitorId')['dt'].shift(-1))
    df['prev_revisit_time'] = df['prev_revisit_time'].astype(np.int64) // 1e9 // 60 // 60
    df.drop('dt', axis=1, inplace=True)
    
def browser_mapping(x):
    if type(x) == float and math.isnan(x):
        return 'NULL'
    
    x = x.lower()
    browsers = ['chrome','safari','firefox','internet explorer','edge','opera','coc coc','maxthon','iron']
    if x in browsers:
        return x
    elif  ('android' in x) or ('samsung' in x) or ('mini' in x) or ('iphone' in x) or ('in-app' in x) or ('playstation' in x):
        return 'mobile browser'
    elif  ('mozilla' in x) or ('chrome' in x) or ('blackberry' in x) or ('nokia' in x) or ('browser' in x) or ('amazon' in x):
        return 'mobile browser'
    elif  ('lunascape' in x) or ('netscape' in x) or ('blackberry' in x) or ('konqueror' in x) or ('puffin' in x) or ('amazon' in x):
        return 'mobile browser'
    elif '(not set)' in x:
        return x
    else:
        return 'others'
    
def adcontents_mapping(x):
    if type(x) == float and math.isnan(x):
        return 'NULL'
    
    x = x.lower()
    if  ('google' in x):
        return 'google'
    elif  ('placement' in x) | ('placememnt' in x):
        return 'placement'
    elif '(not set)' in x or 'nan' in x:
        return x
    elif 'ad' in x:
        return 'ad'
    else:
        return 'others'

def campaign_mapping(x):
    if type(x) == float and math.isnan(x):
        return 'NULL'
    
    x = x.lower()
    if '(not set)' in x:
        return '(not set)'
    elif 'dynamic search' in x:
        return 'dynamic search'
    elif ('accessories' in x) or ('bag' in x) or ('drinkware' in x):
        return 'accessories'
    elif ('data share' in x) or ('music' in x) or ('movie' in x) or ('media' in x):
        return 'data share'
    elif ('electronic' in x) or ('mobile' in x) or ('tech' in x) or ('network' in x):
        return 'electronics'
    elif ('retail' in x) or ('shop' in x):
        return 'retail'
    elif ('apparel' in x) or ('lifestyle' in x) or ('sports' in x):
        return 'apparel'
    else:
        return 'others'
        
def source_mapping(x):
    if type(x) == float and math.isnan(x):
        return 'NULL'
    
    x = x.lower()
    if  ('google' in x):
        return 'google'
    elif  ('youtube' in x):
        return 'youtube'
    elif '(not set)' in x or 'nan' in x:
        return x
    elif 'yahoo' in x:
        return 'yahoo'
    elif 'facebook' in x:
        return 'facebook'
    elif 'reddit' in x:
        return 'reddit'
    elif 'bing' in x:
        return 'bing'
    elif 'quora' in x:
        return 'quora'
    elif 'outlook' in x:
        return 'outlook'
    elif 'linkedin' in x:
        return 'linkedin'
    elif 'pinterest' in x:
        return 'pinterest'
    elif 'ask' in x:
        return 'ask'
    elif 'siliconvalley' in x:
        return 'siliconvalley'
    elif 'lunametrics' in x:
        return 'lunametrics'
    elif 'amazon' in x:
        return 'amazon'
    elif 'mysearch' in x:
        return 'mysearch'
    elif 'qiita' in x:
        return 'qiita'
    elif 'messenger' in x:
        return 'messenger'
    elif 'twitter' in x:
        return 'twitter'
    elif 't.co' in x:
        return 't.co'
    elif 'vk.com' in x:
        return 'vk.com'
    elif 'search' in x:
        return 'search'
    elif 'edu' in x:
        return 'edu'
    elif 'mail' in x:
        return 'mail'
    elif 'ad' in x:
        return 'ad'
    elif 'golang' in x:
        return 'golang'
    elif 'direct' in x:
        return 'direct'
    elif 'dealspotr' in x:
        return 'dealspotr'
    elif 'sashihara' in x:
        return 'sashihara'
    elif 'phandroid' in x:
        return 'phandroid'
    elif 'baidu' in x:
        return 'baidu'
    elif 'mdn' in x:
        return 'mdn'
    elif 'duckduckgo' in x:
        return 'duckduckgo'
    elif 'seroundtable' in x:
        return 'seroundtable'
    elif 'metrics' in x:
        return 'metrics'
    elif 'sogou' in x:
        return 'sogou'
    elif 'businessinsider' in x:
        return 'businessinsider'
    elif 'github' in x:
        return 'github'
    elif 'gophergala' in x:
        return 'gophergala'
    elif 'yandex' in x:
        return 'yandex'
    elif 'msn' in x:
        return 'msn'
    elif 'dfa' in x:
        return 'dfa'
    elif '(not set)' in x:
        return '(not set)'
    elif 'feedly' in x:
        return 'feedly'
    elif 'arstechnica' in x:
        return 'arstechnica'
    elif 'squishable' in x:
        return 'squishable'
    elif 'flipboard' in x:
        return 'flipboard'
    elif 't-online.de' in x:
        return 't-online.de'
    elif 'sm.cn' in x:
        return 'sm.cn'
    elif 'wow' in x:
        return 'wow'
    elif 'baidu' in x:
        return 'baidu'
    elif 'partners' in x:
        return 'partners'
    else:
        return 'others'