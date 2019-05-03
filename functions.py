# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:44:15 2019

@author: CAT
"""
import pandas as pd

def count_unique(df, cols):
    for col in cols:
        print('\n' + 'For column ' + col)
        print(df[col].value_counts())

def fix_data(df, rename_d):
    df = df[~df.isin(['Result', 'Overall Result']).any(axis=1)]
    dfx = df.rename(columns=rename_d)
    return dfx

def fill_month(df1, start_date, end_date, freq):
    month_dict={'': 0,
     'jan': 1,
     'feb': 2,
     'mar': 3,
     'apr': 4,
     'mai': 5,
     'jun': 6,
     'jul': 7,
     'aug': 8,
     'sep': 9,
     'okt': 10,
     'nov': 11,
     'des': 12}
     
    dates = df1.date.astype(str).str.split('.',expand=True)
    dates.iloc[:,-2]= dates.iloc[:,-2].replace(month_dict)
    df1['date'] = dates.iloc[:,-2].astype(str) + '/' + dates.iloc[:,-1].astype(str).map(lambda x: str(x)[-2:])
    
    #___reformat the date columns
    df1['date'] = pd.to_datetime(df1['date'], format='%m/%y')
    
    #create all months in the time range
    all_months=pd.date_range(start=start_date, end=end_date, freq=freq)
    id_list=list(df1['id'].unique())
    
    #ful index with all combinations of date and id
    full_index = pd.MultiIndex.from_product([id_list, all_months],names=['id', 'date'])
    #create a new df, filling the nan values based on the previous records 
    #!!!important that set_index and full index have the same rekkefølge
    df_t1=df1.set_index(['id','date']).reindex(full_index).reset_index()
    return df_t1



def id_diff(df1, df2):
    id_list=list(df1['id'].unique())
    id_list2=list(df2['id'].unique())
    return len([x for x in id_list2 if x not in id_list])


