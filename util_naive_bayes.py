import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


import json
import datetime as dt

from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
  
scaler = None

min_date = dt.date(year=2016, month=8, day=1)

common_word = set(['store', ' stores', 'storre', 'shop', 'shops', 'buy', 'item', 'items','co', 'com',
                              'http', 'https', 'www', 'httpwww', 'httpswww', 'sale', 'sales', 'price', 'online'])

wordls_to_replace_in_keyword = {'merchandise':'merch',
                                }
cols_to_drop = ['socialEngagementType', 
                'traffic_adclick_criteriaParameters', 
                'totals_visits', 'traffic_medium',
                
                'geonet_networkDomain', 
                'geonet_city', 
                'geonet_metro', 
                'geonet_region', 
                
#                'traffic_adContent', 
#                'traffic_campaign', 
#                'traffic_keyword', 
                'traffic_referralPath', 
                'traffic_source',
                
#                'traffic_adclick_adNetworkType',
                'traffic_adclick_gclId',
#                'traffic_adclick_page',
#                'traffic_adclick_slot',
                ]
  
cols_to_fill_zero = ['totals_bounces', 'totals_newVisits', 'totals_pageviews', 
                    'traffic_adclick_page']

keep_browsers = ['Chrome', 'Safari', 'Firefox', 'Internet Explorer', 'Edge', 'Android Webview', 'Safari (in-app)', 'Opera Mini',
                 'Opera', 'UC Browser', 'YaBrowser', 'Coc Coc', 'Amazon Silk', 'Android Browser', 'Mozilla Compatible Agent',
                 'MRCHROME', 'Maxthon', 'BlackBerry', 'Nintendo Browser']

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
                
#                'geonet_networkDomain', 
#                'geonet_city', 
#                'geonet_metro', 
#                'geonet_region', 
                'geonet_continent', 
                'geonet_subContinent', 
                'geonet_country',
                
#                'traffic_adContent', 
#                'traffic_campaign', 
                'traffic_keyword', 
#                'traffic_referralPath', 
#                'traffic_source',
                
                'traffic_adclick_adNetworkType',
#                'traffic_adclick_gclId',
                'traffic_adclick_page',
                'traffic_adclick_slot',
                
#                'weekday', 'month', 'quater'
                ]

def clean_json_blobs(data, is_train=False):
    # device 
    device = [json.loads(data.loc[i, 'device']) for i in data.index]
    data.drop('device', axis=1, inplace=True)
    device = pd.DataFrame.from_records(device)
    device = device[['browser', 'deviceCategory', 'isMobile', 'operatingSystem']]
    device.columns = ['device_' + col for col in device.columns]
  
    # geo_network
    geo_network = [json.loads(data.loc[i, 'geoNetwork']) for i in data.index]
    data.drop('geoNetwork', axis=1, inplace=True)
    geo_network = pd.DataFrame.from_records(geo_network)
    geo_network = geo_network[['city', 'continent', 'country', 'metro', 'networkDomain', 'region', 'subContinent']]
    geo_network.columns = ['geonet_' + col for col in geo_network.columns]
  
    # traffic_source
    traffic_source = [json.loads(data.loc[i, 'trafficSource']) for i in data.index]
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


def count_new_users(values):
#    print(values)
    return 1
#    count = 0
#    for i in range(len(values[0])):
#        if values[0,i] == 1:
#            count += values[1,i]
#    return count
    
def plot_data_2d(x, y=None):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(x)
    
    pca = PCA(n_components = 2)
    pcs = pca.fit_transform(scaled_data) 
    if y is None:
        plt.scatter(pcs[:,0], pcs[:,1], cmap = 'viridis')
    else:
        plt.scatter(pcs[:,0], pcs[:,1], c=y, cmap = 'viridis')
        
        
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
