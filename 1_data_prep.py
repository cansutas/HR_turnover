# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:27:08 2019

@author: CAT
"""

import os
default_path='C:/Users/CAT/OneDrive - Equinor/Projects/Project Strech/Project v.1'
os.chdir(default_path)

import pandas as pd
import numpy as np
from functools import reduce
from functions import fill_month
from functions import id_diff
from functions import count_unique


pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199

#data showing who left the company
leav= pd.read_csv("data/raw/Leav_18.csv", sep=';')
#main dataframe
main= pd.read_csv("data/raw/full_data_monthly_18.csv", sep=';', encoding='latin-1')
#additional travel data
trv= pd.read_csv('data/raw/travel_costs_18.csv', sep=';', encoding='latin-1')
#additional timewriting data
absence= pd.read_csv('data/raw/time_absence_18.csv', sep=';', encoding='latin-1')
attend= pd.read_csv('data/raw/time_attendance_18.csv', sep=';', encoding='latin-1')
#df with country-nation pairs
count_dict= pd.read_csv('data/supporting/country dict.csv', header=None,index_col=0,squeeze=True, encoding='latin-1').to_dict()

#____________________RENAME and CLEAN COLUMNS 
rename_dict= {'Cal. year / month': 'date' , 
             'Employee Equinor': 'id',
             'Employee Equinor.1':'name',
             'Length of service':'service_length',
             'Age in Years':'age',
             'Host Country':'host_country',
             'Nationality': 'nation',
             'Manager (ref. zansnr':'manager_id',
             'Manager (ref. zansnr.1':'manager',
             'Chief Position':'is_manager',
             'Orgunit BA':'BA',
             'Organizational level':'org_level',
             'Discipline.1':'discipline_id',
             'Discipline':'discipline_id',
             'Moves for Permanent employees (from)': 'Leavers', 
             'Branch of study':'study',
             'Process Network':'network', 
             'Certificate':'certif'
             }

leav = leav.rename(columns=rename_dict)
main=main.rename(columns=rename_dict)
attend=attend.rename(columns={'Cal. year / month': 'date' , 'Hours':'attend'})
absence=absence.rename(columns={'Cal. year / month': 'date' , 'Hours':'absence'})
#xtra_info.columns
leav = leav[~leav.isin(['Result', 'Overall Result']).any(axis=1)]
main = main[~main.isin(['Result', 'Overall Result']).any(axis=1)]
absence = absence[~absence.isin(['Result', 'Overall Result']).any(axis=1)]
attend = attend[~attend.isin(['Result', 'Overall Result']).any(axis=1)]
trv = trv[~trv.isin(['Result', 'Overall Result']).any(axis=1)]

main.drop(['Unnamed: 16'], axis=1, inplace=True)
id_list= list(main['id'].unique())
leaver_list= list(leav['id'].unique())

absence['absence'] = pd.to_numeric(absence['absence'].astype('str').apply(lambda x: x.replace(',', '.'))).round()
attend['attend'] = pd.to_numeric(attend['attend'].astype('str').apply(lambda x: x.replace(',', '.'))).round()
trv['travel_avg'] = pd.to_numeric(trv['travel_avg'].astype('str').apply(lambda x: x.replace(',', '.'))).round()

#_____________________CONCENATE__________________________
#merge on id and name using outer to see if there are points from the dfs that doesn't match 
#we remove the ids that weren't in the xtra_info afterwards
df=pd.DataFrame.merge(main,leav, on=['id', 'name'], how='outer')
df['Leavers']= df['Leavers'].replace(np.nan, 0) #If wasn't in the leavers dataset, then not leaver

#check if there were data from leav df that weren't found in the xtr_info
#d1=[value for value in leaver_list if value in id_list]

#keep only the data where id includes the ids from the xtra info
df=df[df['id'].isin(id_list)] #this drops all the missing

#_________COMPLETE MISSING MONTH DATA________________________
df = fill_month(df, '2017-12-01', '2018-12-31', 'MS')
df.sort_values(by=['id', 'date'], inplace=True) 
df=df.groupby(['id']).ffill()
df=df.bfill()

absence= fill_month(absence, '2017-12-01', '2018-12-31', 'MS')
attend= fill_month(attend, '2017-12-01', '2018-12-31', 'MS')

attend['attend']= attend['attend'].replace(np.nan, 0) #If wasn't in the leavers dataset, then not leaver
absence['absence']= absence['absence'].replace(np.nan, 0) #If wasn't in the leavers dataset, then not leaver

absence=absence.drop(['name'], axis=1)
attend=attend.drop(['name'], axis=1)
trv=trv.drop(['name'], axis=1)
#df.isnull().sum()
#______________TIME AND TRAVEL
dfs=[df, absence, attend, trv]
for dfx in dfs:
    dfx['id']=pd.to_numeric(dfx['id'])

df1=pd.DataFrame.merge(df,trv, on=['id'], how='outer')
df1.isnull().sum()
#merge multiple datasets
dfs=[df1, absence, attend]
df_final = reduce(lambda left,right: pd.merge(left,right,on=['id', 'date'], how='outer'), dfs)

#choose only the ids that were in the main dataset
df=df_final[df_final['id'].isin([int(x) for x in id_list])] 

#d5=[value for value in leaver_list_num if value in list(df_final['id'].unique())]
df.isnull().sum() #the ids from time
#see if everyone has 14 rows
df.groupby('id')['id'].count().unique()
#_______________filling forward and backfward for the missin months
df.info()

#miss=df[df.isnull().any(axis=1)]
#miss['Leavers'].sum()/13

#__________________LOOK AT THE VALUES FOR EACH CATEFORICAL
#df.columns
#df.dtypes
#Frequency table
#cat_cols = ['host_country', 'Band', 'is_manager', 'pro_net', 'Gender', 
#            'nation', 'org_level', 'Leavers', 'pro_net_id', 'certif']
#creates a list of unique values for all categorical variables
#count_unique(df, cat_cols)

#Not assigned is nan coding
#country manager # means no

#___________________RECODING VARIABLES______________________________
#___give numerical values to the band and org level 
band_dict={'PROF':3,'PRIN PROF':4,'LEAD PROF':5,'ASSOCIATE':2,'OPER & SUPP':1,
           'MANAGER':6, 'EXECUTIVE':7, 'SR EXEC':8}

org_lev_dict={'Dept level 5':6, 'Sector Level':5, 'Dept level 6':7, 'Business Unit':4,
       'Business Cluster':3, 'Business Area':2, 'Corporate':1, 'Dept level 7':8,
       'Dept level 8':9}

df['Band']= df['Band'].replace(band_dict)
df['org_level']= df['org_level'].replace(org_lev_dict)

#___fix the categorical variables to binary
df['is_manager']= df['is_manager'].replace({'#':0, 'X':1})
df['Gender']=df['Gender'].replace(2, 0)

#________________HANDLING MISSING VALUES
#count the number of Not assigned
#for col in df.columns:
#    if df[col].dtype == object:
#        count = 0
#        count = [count + 1 for x in df[col] if x == 'Not assigned']
#        print(col + ' ' + str(sum(count)))

#df.info()
#___convert different type of NaN to np.nan
df= df.replace('Not assigned',np.nan) 
#df['manager_id']= df['manager_id'].replace('#',0)
df= df.replace('#',np.nan)

             
df.describe()
df['attend']= df['attend'].replace(np.nan, 0) 
df['absence']= df['absence'].replace(np.nan, 0) #If wasn't in the leavers dataset, then not leaver
df['travel_avg']= df['travel_avg'].replace(np.nan, 0) 
df['service_length']= df['service_length'].replace(np.nan, 0)     

certif_dict = dict.fromkeys(['DD', 'CE', 'DE', 'BD', 'ED', 'BB', 'DG', 'EJ', 'DH', 'BC', 'BA',
       'BG', 'BH', 'EE', 'EC', 'CB', 'DF',  'DI', 'CA'], np.nan)
certif_dict['99']=np.nan 
df['certif']= df['certif'].replace(certif_dict)
df.sort_values(by=['id', 'date'], inplace=True) 
df[['id', 'certif']]=df[['id', 'certif']].groupby(['id']).ffill()
df[['id','certif']]=df[['id', 'certif']].groupby(['id']).bfill()

cont_var=['service_length', 'manager_id', 'certif'] #numeric variables
for x in cont_var:
    df[x]=pd.to_numeric(df[x], errors='raise')
    
miss=df[df.isnull().any(axis=1)]
miss['Leavers'].sum()/13
#df.info()    
#_______fill some missing values
#fill missing org_level based on the average org_level of the same band
miss_ids= list(df['id'][df.isnull().sum(axis=1)>4].unique())
df=df[~df['id'].isin(miss_ids)] 
df.isnull().sum() #

df['Band'].fillna(df.groupby(['org_level', 'Age'])['Band'].transform('mean').round(), inplace=True)
df['org_level'].fillna(df.groupby('Band')['org_level'].transform('mean').round(), inplace=True)

df['certif'].fillna(df.groupby(['Band', 'is_manager'])['certif'].transform('median').round(), inplace=True)

df.drop(columns='manager', inplace=True)
df.sort_values(by=['id', 'date'], inplace=True) 
df[['id', 'manager_id']]=df[['id', 'manager_id']].groupby(['id']).ffill()
df[['id','manager_id']]=df[['id', 'manager_id']].groupby(['id']).bfill()


df['manager_id'].fillna(df.groupby(['Band', 'BA', 'org_level'])['manager_id'].transform('median'), inplace=True)

miss=df[df.isnull().any(axis=1)]
miss['Leavers'].sum()/13

miss_ids= list(df['id'][df.isnull().sum(axis=1)>0].unique())
df=df[~df['id'].isin(miss_ids)] 
#df.isnull().sum()

#do some exlor to decide how to group for means of certificate
#import matplotlib.pyplot as plt
#%matplotlib inline
#cat_cols = ['Band', 'is_manager', 'Gender', 'org_level']
#for col in cat_cols:
#    means=df.groupby(col)['certif'].mean()
#    means.plot.bar(color = 'green')
#    plt.xlabel(col)
#    plt.ylabel('Leaver Percentage')
##    plt.savefig(('plots/'+'leaver_perc_'+col))
#    plt.show()

#___________________FEATURE ENGINEERING_____________
#______create historical data for host country and discipline
#count number of countries/disciplines each person have in their data
df['host_count']=df.groupby('id')['host_country'].transform('nunique')
df['network_count']=df.groupby('id')['network'].transform('nunique')

#___create promoted
df.sort_values(by=['id', 'date'], inplace=True) 
df['promotion_1'] = df['Band'].diff() #create new variable with the difference of promotion variable latter row-previous row
mask = df.id != df.id.shift(1) #create mask where ids are different from row to row (new person) 
df['promotion_1'][mask] = 0 
#copy the data to all rows promotion column for each person
df['promotion'] = df.groupby(['id'])['promotion_1'].apply(lambda x: x.cumsum())
df.drop(columns=['promotion_1'], inplace=True)
#df.isnull().sum()

#_____create a new variable called expat based on the nation and host country information
df['nation'].head()
df['host_country'].head()
#_all lower case
df.nation = df.nation.astype(str).apply(lambda x: x.lower())
df.host_country = df.host_country.astype(str).apply(lambda x: x.lower())
count_dict= {k.lower(): v.lower() for k, v in count_dict.items()}

df['nation']= df['nation'].replace({'nan':np.nan,'kasachstani':'kazakh',
  'columbian':'colombian', 'brithish':'british','white-russian':'belarusian',
  'f. trinidad & t':'trinidadian', 'ghanian':'ghanaian', 'neth.antillean':'antillean'})

df['host_country']= df['host_country'].replace({'nan':np.nan,'brasil':'brazil','great britain': 'united kingdom', 'russian fed.': 'russia'})
#_create the home country column
df['home_country']= df['nation'].map(count_dict)
df['home_country'].head()

#_create the expat columns
df.loc[df['home_country']!=df['host_country'], 'expat'] = 1
df.loc[df['home_country']==df['host_country'], 'expat'] = 0
#change some mismatch because of nan into nans 
df['expat'] =np.where(((df['host_country'].isnull()) | (df['home_country'].isnull())), np.nan, df['expat'] )

#___create team size
#team size of people, based on shared manager, so managers are only counted 
#in the team of their manager
df['team_size']= df.groupby('manager_id')['id'].transform('nunique') #this wont be accurate because of filling the missing values

df['is_norsk']= np.where(df['nation']=='norwegian', 1, 0)
df['in_norge']= np.where(df['host_country']=='norway', 1, 0)

df['attend_avg']=df.groupby(['id'])['attend'].transform('mean').round()
df['absence_avg']=df.groupby(['id'])['absence'].transform('mean').round()

#see the minimum row number someone has
df.groupby('id')['id'].count().unique()
miss=df[df.isnull().any(axis=1)]
#__choose the most recent 6 data points for each person 
#to make sure everyone has the same amount of data

#df.sort_values(by=['id', 'date'], inplace=True) 
#idx= list(df.groupby('id').tail(6).index)
#df_recent= df[df.index.isin(idx)]
#
#df_recent.groupby('id')['id'].count().unique()
#df_recent.to_csv('data/processed/6months_df_filled.csv', encoding='utf-8', index=False)
#__________REMOVE MULTIPLE MONTH ROWS____________________
manager_ids=list(df['manager_id'].unique())
managers=df[df['id'].isin(manager_ids)]

#use the managers dataset to get info about each persons manager
df_man = pd.merge(df, managers.set_index(['date','id']), left_on=['date', 'manager_id'], how='left', right_index=True, suffixes=('', '_man'))

df_man.dropna(inplace=True)

#Df without the manager info

#df['man_direct_reports'] = df['id'].map(df['manager_id'].value_counts()) 
#df['man_leaver_count'] = df.groupby('manager_id')['Leavers'].transform('sum')
#managers=df[df['id'].isin(manager_ids)]
#managers.to_csv('data/processed/managers.csv', encoding='utf-8', index=False)
#df.drop(columns=['man_direct_reports', 'man_leaver_count'], inplace=True) 

#df_man['Leavers'].sum()
#narrow the dataset to have single line per person

idx = df_man.groupby('id')['date'].transform(max) == df_man['date']
df_narrow = df_man[idx]
df_narrow=df_narrow.drop(columns=['attend', 'absence'])
#create two variables special for managers data set
df_narrow['direct_reports'] = df_narrow['id'].map(df_man['manager_id'].value_counts()) 
df_narrow['leaver_count'] = df_narrow.groupby('manager_id')['Leavers'].transform('sum')

#choose only the managers from the narrow data set
managers_man=df_narrow[df_narrow['id'].isin(manager_ids)]
#save
managers_man.to_csv('data/processed/managers_man.csv', encoding='utf-8', index=False)
#drop the manager special columns
df_narrow=df_narrow.drop(['direct_reports', 'leaver_count'], axis=1)

Leavers_single= df_narrow[df_narrow['Leavers']==1]
Leavers_full= df[df['Leavers']==1]
Leavers_single.to_csv('data/processed/leavers_single.csv', encoding='utf-8', index=False)
Leavers_full.to_csv('data/processed/leavers_full.csv', encoding='utf-8', index=False)


df_narrow.to_csv('data/processed/single_df_filled.csv', encoding='utf-8', index=False)
df_man.to_csv('data/processed/full_df_filled.csv', encoding='utf-8', index=False)



#df_narrow['man_Band']=df_narrow['manager_id'].map(df_narrow.set_index('id')['Band'])
#df_narrow['man_age']=df_narrow['manager_id'].map(df_narrow.set_index('id')['age'])