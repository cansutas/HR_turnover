# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 17:27:52 2019

@author: CAT
"""
#________________________DATA EXPLORATION________________________

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

default_path='C:/Users/CAT/OneDrive - Equinor/Projects/Project Strech/Project v.1'
os.chdir(default_path)

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199

df= pd.read_csv("data/processed/single_df_filled.csv")
man= pd.read_csv('data/processed/manAgers_man.csv')
df.info()

#Frequency table
cat_cols = ['host_count', 'network_count', 'Band', 'is_manager', 'Gender', 'promotion', 'expat', 'org_level', 'network', 'is_norsk', 'in_norge']
num_cols = ['Age', 'service_length', 'absence_avg', 'attend_avg', 'travel_avg', 'certif', 'team_size']

for col in cat_cols:
    print('\n' + 'For column ' + col)
    print(df[col].value_counts())

#_______________________________________FREQUENCY EXPLORATION WITH GRAPHS

#frequency bars FOR CATEGORICAL
for col in cat_cols:
    fig = plt.figure(figsize=(6,6)) # define plot area
    ax = fig.gca() # define axis    
    counts = df[col].value_counts().sort_index() # find the counts for each unique category
    counts.plot.bar(ax = ax, color = 'blue') # Use the plot.bar method on the counts data frame
    ax.set_title('Number of employees by ' + col) # Give the plot a main title
    ax.set_xlabel(col) # Set text for the x axis
    ax.set_ylabel('Number of employees')# Set text for y axis
    plt.show()
    fig.savefig(('plots/'+'freq_'+col))


    ax.set_xticks(x)    
    ax.set_xticklabels(x_labels, rotation=90) #set the labels and rotate them 90 deg.

    
#histograms FOR NUMERICAL
for col in num_cols:
    fig = plt.figure(figsize=(6,6)) # define plot area
    ax = fig.gca() # define axis    
    df[col].plot.hist(ax = ax, bins = 10) # Use the plot.hist method on subset of the data frame
    ax.set_title('Histogram of ' + col) # Give the plot a main title
    ax.set_xlabel(col) # Set text for the x axis
    ax.set_ylabel('Number of employees')# Set text for y axis
    plt.show()
  #split by Leavers
#for col in num_cols:
#    grid1 = sns.FacetGrid(df, col='Leavers')
#    grid1.map(plt.hist, col, alpha=.7)


df.groupby('Leavers').mean()
df['Leavers'].value_counts()
#df.groupby('BC')['Leavers'].mean().sort_values()
#df.groupby('Band')['Leavers'].mean()
#df.groupby('expat')['Leavers'].mean()
#df.groupby('host_country')['Leavers'].mean().sort_values()
#df.groupby('org_level')['Leavers'].mean()

#df[df['expat']==1].groupby('nation')['Leavers'].mean().sort_values()
###explore the means a bit deeper
#mostly SR Exec and associate leaving

#___Age
df[df['Leavers']==1].groupby('Age')['Leavers'].count().plot()

##___discipline
#d=df.groupby('network')['Leavers'].mean()
#d=d[d>0.1]
#d_l= df[df['Leavers']==1].groupby('discipline')['Leavers'].count()
#discip_size= df.groupby('discipline')['Leavers'].count()
#d_df=pd.concat([d,d_l, discip_size],axis=1)
#d_df.columns=['percentAge', 'count','team_size']
#d_df=d_df.dropna()
#print(d_df.sort_values(by='percentAge'))

#def plot_scatter(df, cols, col_y = 'service_length'):
#    for col in cols:
#        fig = plt.figure(figsize=(7,6)) # define plot area
#        ax = fig.gca() # define axis   
#        df.plot.scatter(x = col, y = col_y, ax = ax)
#        ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col) # Give the plot a main title
#        ax.set_xlabel(col) # Set text for the x axis
#        ax.set_ylabel(col_y)# Set text for y axis
#        plt.show()
#
#________________VISUALISATION RELATED TO THE MEASURE______________
#leaver percentAge per category level
for col in cat_cols:
    means=df.groupby(col)['Leavers'].mean()
    means.plot.bar(color = 'green')
    plt.xlabel(col)
    plt.ylabel('Leaver PercentAge')
    plt.savefig(('plots/'+'leaver_perc_'+col), bbox_inches = "tight")
    plt.figure()
    plt.show()

%matplotlib inline
'org_level', 'Gender', 'Band',     
cols=['is_manager', 'expat', 'is_norsk', 'in_norge']
for col in cols:    
    means=df.groupby(col)['Leavers'].mean()
    means.plot.bar(color = 'green')
    plt.xlabel(col)
    plt.ylabel('Leaver PercentAge')  
    plt.set_xticklabels(['No', 'Yes'], rotation=90) #set the labels and rotate them 90 deg.
#    plt.savefig(('plots/'+'leaver_perc_'+col))
    plt.figure()
    plt.show()    
    
    
    
#kernel density for num variables
for col in num_cols:
    sns.set_style("whitegrid")
    sns.jointplot(col, 'service_length', data=df, kind='kde')
    plt.xlabel(col) # Set text for the x axis
    plt.ylabel('service_length')# Set text for y axis
    plt.savefig(('plots/'+'kde_service_len'+col))

#CATEGORICAL VERSION 1
#bar chart color coded by Leavers, bars side by side
#for col in cat_cols:
#    pd.crosstab(df[col],df['Leavers']).plot(kind='bar')
#    plt.title('Count of Employees by ' + col)
#    plt.xlabel(col)
#    plt.ylabel('Count of Employees')
#    plt.savefig(('plots/'+'leaver_count'+col))
#    plt.show()

#CATEGORICAL VERSION 2
#stackked bar chart color coded by Leavers
for col in cat_cols:
    table=pd.crosstab(df[col],df['Leavers'])
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('PercentAge of Leavers by ' + col)
    plt.xlabel(col)
    plt.ylabel('Proportion of Employees')
    plt.savefig(('plots/'+'leaver_prop_by_'+col))

#CATEGORICAL VERSION 3
#it show 2 seperate graphs for leaver 0 and 1 side by side. The y axis is adjusted so 
    #its visually easier to see the proportions


#df['dummy'] = np.ones(shape = df.shape[0])
#for col in cat_cols:
#    counts = df[['dummy', 'Leavers', col]].groupby(['Leavers', col], as_index = False).count()
#    _ = plt.figure(figsize = (10,4))
#    plt.subplot(1, 2, 1)
#   
#    temp = counts[counts['Leavers'] == 0][[col, 'dummy']]
#    plt.bar(temp[col], temp.dummy)
#    plt.xticks(rotation=90)
#    plt.title('Counts for ' + col + '\n Leavers')
#    plt.ylabel('count')
#    plt.subplot(1, 2, 2)
#   
#    temp = counts[counts['Leavers'] == 1][[col, 'dummy']]
#    plt.bar(temp[col], temp.dummy)
#    plt.xticks(rotation=90)
#    plt.title('Counts for ' + col + '\n Stayers')
#    plt.ylabel('count')
#    plt.show()



#___visualise numerical 
#PLOT FUNCTIONS 
for col in num_cols:
    sns.set_style("whitegrid")
    sns.boxplot('Leavers', col, data=df)
    plt.xlabel('Leavers') # Set text for the x axis
    plt.ylabel(col)# Set text for y axis
    plt.savefig(('plots/'+'leaver_box_by_'+col))
    plt.figure()
    plt.show()
    
for col in num_cols:
    sns.set_style("whitegrid")
    sns.violinplot('Leavers', col, data=df)
    plt.xlabel('Leavers') # Set text for the x axis
    plt.ylabel(col)# Set text for y axis
    plt.show()
    


##Cond plots 
#scatter: service len by another numerical in num_col
#split by Band and by Leavers
#color: manAger id 
num_cols = ['Age', 'absence_avg', 'attend_avg', 'travel_avg', 'certif']  
band_dict={'PROF':3,'PRIN PROF':4,'LEAD PROF':5,'ASSOCIATE':2,'OPER & SUPP':1,
           'MANAgeR':6, 'EXECUTIVE':7, 'SR EXEC':8}

for col in num_cols:
    g = sns.FacetGrid(df, col="Band", row = 'Leavers', 
                  hue="is_manager", palette="Set2", margin_titles=True)
    g.map(sns.regplot, col, "service_length", fit_reg = False)
    for ax, title in zip(g.axes.flat, ['OPER & SUPP','ASSOCIATE','PROF','PRIN PROF','LEAD PROF',
           'MANAgeR', 'EXECUTIVE', 'SR EXEC']):
        ax.set_title(title)
    plt.savefig(('plots/'+'scatter_by_'+col), dpi=900)


#______________________________________MANAGER EXPLORATION
cat_cols=['network', 'host_country', 'Band', 'Gender', 'nation', 'org_level', 'host_count', 'network_count', 'promotion', 'is_norsk', 'in_norge']
num_cols=['Age', 'service_length', 'certif', 'time_avg', 'travel_avg', 'direct_reports', 'leaver_count']

%matplotlib inline
for col in num_cols:
    fig = plt.figure(figsize=(6,6)) # define plot area
    ax = fig.gca() # define axis    
    man[col].plot.hist(ax = ax, bins = 20) # Use the plot.hist method on subset of the data frame
    ax.set_title('Histogram of ' + col) # Give the plot a main title
    ax.set_xlabel(col) # Set text for the x axis
    ax.set_ylabel('Number of managers')# Set text for y axis
    plt.show()


#leaver average per category level for each manager
for col in cat_cols:
    count=man.groupby(col)['leaver_count'].mean()
    count.plot.bar(color = 'green')
    plt.xlabel(col)
    plt.ylabel('Average of Direct Reports Left')
#   plt.savefig(('plots/'+'leaver_perc_'+col))
    plt.figure()
    plt.show()

#host country, nation, promotion, network count, host country count, org level, band and network seems to make a difference
    

#kernel density for num variables
for col in num_cols:
    sns.set_style("whitegrid")
    sns.jointplot(col, 'leaver_count', data=man, kind='kde')
    plt.xlabel(col) # Set text for the x axis
    plt.ylabel('leaver_count')# Set text for y axis
#    plt.savefig(('plots/'+'kde_service_len'+col))

##Cond scatter plots 
#scatter: sleaver_count by another numerical in num_col
#split by Band and by Leavers
#hue: expat
num_cols = ['Age', 'time_avg', 'travel_avg', 'certif', 'service_length']  
%matplotlib auto
for col in num_cols:
    g = sns.FacetGrid(man, hue="expat", palette="Set1", margin_titles=True, legend_out=True)
    g.map(sns.regplot, col, "leaver_count", fit_reg = False, edgecolor="w", s='Age').add_legend()

col='service_length'
sns.scatterplot(x="leaver_count", y=col, hue="expat", size='Age', sizes=(20, 100), data=man, y_jitter=5);
            


#    plt.savefig(('plots/'+'scatter_by_'+col), dpi=900)


#___________________________EMPLOYEES MANAGERS 

#ARCHIVE
#_______________combine Leavers and numerical variables

#def plot_scatter_shape(df, cols, shape_col = 'Leavers', col_y = 'service_length', alpha = 0.2):
#    shapes = ['+', 'o', 's', 'x', '^'] # pick distinctive shapes
#    unique_cats = df[shape_col].unique()
#    for col in cols: # loop over the columns to plot
#        sns.set_style("whitegrid")
#        for i, cat in enumerate(unique_cats): # loop over the unique categories
#            temp = df[df[shape_col] == cat]
#            sns.regplot(col, col_y, data=temp, marker = shapes[i], label = cat,
#                        scatter_kws={"alpha":alpha}, fit_reg = False, color = 'blue')
#        plt.title('Scatter plot of ' + col_y + ' vs. ' + col) # Give the plot a main title
#        plt.xlabel(col) # Set text for the x axis
#        plt.ylabel(col_y)# Set text for y axis
#        plt.legend()
#        plt.show()
#            
#num_cols = ['Age', 'time_avg', 'travel_avg']
#plot_scatter_shape(df, num_cols)      

##_________color size and shape
#%matplotlib auto
#
##create dictionaries for the legend
#gender_dict={0:'Female', 1:'Male'}
#leaver_dict={0.0:'Stayed', 1:'Left'}

#color=cat variable
#shape=cat variable
#size=num variable
#coly=important num variable
#cols =other num variables

#def plot_scatter_shape_size_col(df, cols, shape_col = 'Gender', size_col = 'Band',
#                            size_mul = 2, color_col = 'Leavers', col_y = 'service_length', alpha = 0.4):
#    shapes = ['+', 'o', 's', 'x', '^'] # pick distinctive shapes
#    colors = ['green', 'red', 'blue', 'mAgenta', 'gray'] # specify distinctive colors
#    unique_cats = df[shape_col].unique()
#    unique_colors = df[color_col].unique()
#    for col in cols: # loop over the columns to plot
#        sns.set_style("whitegrid")
#        for i, cat in enumerate(unique_cats): # loop over the unique categories
#            for j, color in enumerate(unique_colors):
#                temp = df[(df[shape_col] == cat) & (df[color_col] == color)]
#                sns.regplot(col, col_y, data=temp, marker = shapes[i],
#                            scatter_kws={"alpha":alpha, "s":size_mul*temp[size_col]**2}, 
#                            label = (gender_dict[cat] + ' and ' + leaver_dict[color]), fit_reg = False, color = colors[j])
#        plt.title('Scatter plot of ' + col_y + ' vs. ' + col) # Give the plot a main title
#        plt.xlabel(col) # Set text for the x axis
#        plt.ylabel(col_y)# Set text for y axis
#        plt.legend()
#        plt.figure()
#        plt.show()
#
#num_cols = ['Age', 'time_avg', 'travel_avg']       
#plot_scatter_shape_size_col(df, num_cols)     
#
#
#

