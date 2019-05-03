# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:23:22 2019

@author: CAT
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import random
random.seed(42)

default_path='C:/Users/CAT/OneDrive - Equinor/Projects/Project Strech/Project v.1'
os.chdir(default_path)
df_address="single_df_filled"

def data_coding(df_address, drop_cols):
    df= pd.read_csv(('data/processed/' + df_address + '.csv'))
    df_ready= df.drop(columns= drop_cols) # removing attend and absence cause we have the average
    
    #________RECODE VARIABLES
    #give numbers 1 to ... to manager id
#    df_ready.sort_values(by=['manager_id'], inplace=True)
#    df_ready['manager_id']= df_ready['manager_id'].astype('category').cat.codes
    
    #___ONE HOT CODING__________
    cat_vars=['host_country', 'nation', 'host_country_man', 'nation_man']  
    
    for col in cat_vars:
        #create other variable for less then 0.5 percent
        series = pd.value_counts(df_ready[col])
        mask = (series/series.sum() * 100).lt(0.55) #change lt() to change the other criteria
        # To replace df['column'] use np.where I.e 
        df_ready[col] = np.where(df_ready[col].isin(series[mask].index),'Other',df_ready[col])
    
    cat_vars=['host_country', 'nation', 'host_country_man', 'nation_man', 'BA', 'network', 'BA_man', 'network_man']  
    df_coded = pd.get_dummies(df_ready, prefix_sep="_", columns=cat_vars)
    #df_coded.drop(columns=cat_vars, axis=1, inplace=True)
    df_coded.columns.values
    
    ###_______________SOLVE THE INF ISSUE_______-
    df_coded=df_coded.astype('float64')
    
    return df_coded

#cols to drop for the full data
#drop_cols= ['date', 'name', 'home_country', 'home_country_man',
#                                'is_norsk', 'in_norge', 'is_norsk_man', 'in_norge_man', 'name_man', 'is_manager_man', 'attend', 'attend_man',
#                                'absence','absence_man', 'manager_id', 'manager_id_man']
#cols to drop for the single data
drop_cols= ['date', 'name', 'home_country', 'home_country_man',
                                'is_norsk', 'in_norge', 'is_norsk_man', 'in_norge_man', 'name_man', 'is_manager_man', 'manager_id', 'manager_id_man']

df_coded= data_coding(df_address, drop_cols)
df_coded.to_csv(('data/processed/' + df_address + '_coded.csv'), index=False)


##CREATE TRAIN TEST SPLITS
#y is the column name #identify is the id column name
def test_train_manual(main_df, y, identif):

    #divide the dataset into leavers and non-leavers
    y_1=main_df[main_df[y]==1]
    y_0=main_df[main_df[y]==0]
    #create list of unique ids to make sure data for the same id is not found in the both dataset
    y_1_id= list(y_1[identif].unique()) #list of ids for nonleavers
    y_0_id= list(y_0[identif].unique())
    
    #split the leaver and nonleaver ids into 25% and put them together for the test set
    test_id= random.sample(y_1_id, (len(y_1_id)//4)) + random.sample(y_0_id, (len(y_0_id)//4))
    train_id= [x for x in list(main_df[identif].unique()) if x not in test_id] #remaining ids is the train set
    
    #test and train dfs based on the ids selected
    test_df=main_df[main_df[identif].isin(test_id)]
    train_df=main_df[main_df[identif].isin(train_id)]
    
    #split X and y
    test_df.drop(columns=identif, inplace=True)
    train_df.drop(columns=identif, inplace=True)
    
    features=test_df.columns.values.tolist() #features are the same in both
    y=[y]
    X=[i for i in features if i not in y]
    
    X_test=test_df[X]
    X_train=train_df[X]
    y_test=test_df[y].values.ravel()
    y_train=train_df[y].values.ravel()
    return X_test, X_train, y_test, y_train

#main_df=df_coded
#
##divide the dataset into leavers and non-leavers
#y_1=main_df[main_df['Leavers']==1]
#y_0=main_df[main_df['Leavers']==0]
##create list of unique ids to make sure data for the same id is not found in the both dataset
#y_1_id= list(y_1['id'].unique()) #list of ids for nonleavers
#y_0_id= list(y_0['id'].unique())
#
##split the leaver and nonleaver ids into 25% and put them together for the test set
#test_id= random.sample(y_1_id, (len(y_1_id)//4)) + random.sample(y_0_id, (len(y_0_id)//4))
#train_id= [x for x in list(main_df['id'].unique()) if x not in test_id] #remaining ids is the train set
#
##test and train dfs based on the ids selected
#test_df=main_df[main_df['id'].isin(test_id)]
#train_df=main_df[main_df['id'].isin(train_id)]
#
##split X and y
#test_df.drop(columns='id', inplace=True)
#train_df.drop(columns='id', inplace=True)
#
#features=test_df.columns.values.tolist() #features are the same in both
#y=['Leavers']
#X=[i for i in features if i not in y]
#
#X_test=test_df[X]
#X_train=train_df[X]
#y_test=test_df[y]
#y_train=train_df[y]

X_test, X_train, y_test, y_train = test_train_manual(df_coded, 'Leavers', 'id')

def select_features(X_test, X_train, y_test, y_train):
    rf_init = RandomForestClassifier(n_estimators= 400, random_state=42)
    rf_init.fit(X_train, y_train)
    
    print(classification_report(y_test, rf_init.predict(X_test)))
    
    #________________feature importance
    cols=X_test.columns.values.tolist()
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature,
                           importance in zip(cols, list(rf_init.feature_importances_))]
    
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    print(feature_importances)

    # Extract the names of the most important features based on the number of imp features
    important_feature_names = [feature[0]  for feature in feature_importances if feature [1]>0]
    
    #____________Create training and testing sets with only the important features
    imp_X_train = X_train[important_feature_names]
    imp_X_test = X_test[important_feature_names]
    # Sanity check on operations
    print('Important train features shape:', imp_X_train.shape)
    print('Important test features shape:', imp_X_test.shape)
    
    return imp_X_train, imp_X_test, feature_importances

imp_X_train, imp_X_test, feature_importances = select_features(X_test, X_train, y_test, y_train)

df.info()

#ARCHIVE
    

#
#df= pd.read_csv("data/processed/single_df_filled.csv")
#df= pd.read_csv('data/processed/6months_df_filled.csv')
#
#df.columns
#df_ready= df.drop(columns= ['index', 'date', 'id', 'discipline','name_x', 'home_country'])
#df_ready.info()
#df_ready.isnull().sum()
#df_ready[df_ready.dtypes=='object'].names
#
##________RECODE VARIABLES
##give numbers 1 to ... to manager id
#df_ready.sort_values(by=['manager_id'], inplace=True)
#df_ready['manager_id']= df_ready['manager_id'].astype('category').cat.codes
##give numbers 1 to ... to discipline id
#df_ready.sort_values(by=['discipline_id'], inplace=True)
#df_ready['discipline_id']= df_ready['discipline_id'].astype('category').cat.codes
#
##___ONE HOT CODING__________
#cat_vars=['host_country', 'nation']  
#
#for col in cat_vars:
#    #create other variable for less then 0.5 percent
#    series = pd.value_counts(df_ready[col])
#    mask = (series/series.sum() * 100).lt(0.55) #change lt() to change the other criteria
#    # To replace df['column'] use np.where I.e 
#    df_ready[col] = np.where(df_ready[col].isin(series[mask].index),'Other',df_ready[col])
#
#
#df_coded = pd.get_dummies(df_ready, prefix_sep="_", columns=cat_vars)
##df_coded.drop(columns=cat_vars, axis=1, inplace=True)
#df_coded.columns.values
#

#
#
#
#
#df_coded.to_csv('data/processed/df_coded.csv', index=False)