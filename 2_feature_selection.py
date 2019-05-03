# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 08:12:02 2019

@author: CAT
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os

default_path='C:/Users/CAT/OneDrive - Equinor/Projects/Project Strech/Project v.1'
os.chdir(default_path)
df= pd.read_csv("data/processed/full_df_filled_coded.csv")

#____________1)SELECT IMPORTANT FEATURES
main_df=df

def select_features(main_df, num):
    features=main_df.columns.values.tolist()
    y=['Leavers']
    X=[i for i in features if i not in y]
    X=main_df[X]
    y=main_df['Leavers']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=main_df[[y,  )
    rf_init = RandomForestClassifier(n_estimators= 400, random_state=42)
    rf_init.fit(X_train, y_train)
    
    print(classification_report(y_test, rf_init.predict(X_test)))
    
    #________________feature importance
    cols=X.columns.values.tolist()
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
    
    dfs= [imp_X_train, imp_X_test, y_train, y_test]
    df_names= ['imp_X_train', 'imp_X_test', 'y_train', 'y_test']
    
    for df, df_name in zip(dfs, df_names):
        df.to_csv(('data/model/' + df_name + num + '.csv'), index=False)
    return feature_importances

feature_importances = select_features(df, '2')

df.info()

#print('Training Features Shape:', X_train.shape)
#print('Training Labels Shape:', y_train.shape)
#print('Testing Features Shape:', X_test.shape)
#print('Testing Labels Shape:', y_test.shape)


##______Recursive Feature Elimination______________
#from sklearn.feature_selection import RFE
#model = RandomForestClassifier(n_estimators= 1000, random_state=42)
#rfe = RFE(model, 20)
#rfe = rfe.fit(df[X], df[y])
#
#feature_array=np.array(features)
#filtering=np.array(rfe.support_)
#print(sorted(zip(rfe.support_, features)))
#print(sorted(zip(rfe.ranking_,X)))
#
#from itertools import compress
#fil =list(rfe.support_)
#cols=list(compress(features, fil))
#
#X=df[cols]
#y=df['Leavers']
