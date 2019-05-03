# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 12:40:20 2019

@author: CAT
"""
import os
default_path='C:/Users/CAT/OneDrive - Equinor/Projects/Project Strech/Project v.1'
os.chdir(default_path)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.externals import joblib

from sklearn import metrics

df= pd.read_csv("data/processed/full_df_filled_coded.csv")
from data_preprocessing_1 import test_train_manual
from data_preprocessing_1 import select_features
import random

X_test, X_train, y_test, y_train = test_train_manual(df, 'Leavers', 'id')
imp_X_train, imp_X_test, feature_importances = select_features(X_test, X_train, y_test, y_train)

#____________________2) Model with reduced features
# Train the expanded model on only the important features
rf = RandomForestClassifier(n_estimators= 400, random_state=42)
rf.fit(imp_X_train, y_train);

print(classification_report(y_test, rf.predict(imp_X_test)))

#___________________3) Model with up/down sampling
#DOWNSAMPLING
# now let us check in the number of Percentage
count_stay = len(df[df["Leavers"]==0]) 
count_leav = len(df[df["Leavers"]==1]) 
perc_stay = count_stay/(count_stay+count_leav)
print("percentage of stay is",perc_stay*100)
perc_leav= count_leav/(count_stay+count_leav)
print("percentage of leave ",perc_leav*100)

#combine the training X and y 
imp_X_train["Leavers"]= y_train
df_train = imp_X_train.copy().reset_index(drop=True) # for naming 
imp_X_train.drop(columns='Leavers', inplace=True)
print("length of training df",len(df_train))

# for undersampling we need a portion of majority Leavers and will take whole df of minority Leavers

leav_ind= np.array(df_train[df_train.Leavers==1].index)
stay_ind = np.array(df_train[df_train.Leavers==0].index)
#now let us a define a function for make undersample df with different proportion
#different proportion means with different proportion of normal Leaverses of df
def undersample(stay_ind,leav_ind,times):#times denote the normal df = times*fraud df
    stay_ind_undersample = np.array(np.random.choice(stay_ind,(times*count_leav),replace=False)) #choose a number of indices from stay
    undersample_df= np.concatenate([leav_ind,stay_ind_undersample]) #indices for all leavers and some stayers
    undersample_df = df_train.iloc[undersample_df,:] #create the df based on indices
    
    print("the stay proportion is :",len(undersample_df[undersample_df.Leavers==0])/len(undersample_df['Leavers']))
    print("the leav proportion is :",len(undersample_df[undersample_df.Leavers==1])/len(undersample_df['Leavers']))
    print("total number of record in resampled df is:",len(undersample_df['Leavers']))
    #seperate the X and y again
    features=undersample_df.columns.values.tolist()
    y=['Leavers']
    X=[i for i in features if i not in y]
    
    under_X_train=undersample_df[X]
    under_y_train=undersample_df['Leavers']
    return(under_X_train, under_y_train)

#let us train this model using undersample data and test for the whole data test set 
#for i in range(4,8):
#    print("the undersample data for {} proportion".format(i))
#    print()
#    (under_X_train, under_y_train)=undersample(stay_ind,leav_ind,i)
#    print("------------------------------------------------------------")
#    print()
#    print("the model classification for {} proportion".format(i))
#    print()
#    
#    clf=RandomForestClassifier(n_estimators=100)
#    clf.fit(under_X_train, under_y_train)
#    print(classification_report(y_test, clf.predict(imp_X_test)))

#the best performnace is with the 6 proportion
(under_X_train, under_y_train)=undersample(stay_ind,leav_ind,6)

rf.fit(under_X_train, under_y_train)
print(classification_report(y_test, rf.predict(imp_X_test)))

## TRY WITH THE BETTER PARAMETERS IF YOU DID THE PARAMETER SEARCH
rf_imp_para = RandomForestClassifier(n_estimators= 400, 
                                random_state=42, 
                                min_samples_leaf=1, 
                                max_features='sqrt',
                                bootstrap= True
                                )
rf_imp_para.fit(under_X_train, under_y_train)
predictions=rf_imp_para.predict(imp_X_test)
print(classification_report(y_test, predictions))
# Train and Test Accuracy

print("Train Accuracy :: ", accuracy_score(y_train, rf_imp_para.predict(imp_X_train)))
print("Test Accuracy  :: ", accuracy_score(y_test, predictions))
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_test, predictions))
plt.title(all_sample_title, size = 15);
#__________cross validation

from sklearn import model_selection

kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = rf_imp_para
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, under_X_train, under_y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
print(results.std())




#________________EXPORTING THE MODEL
estimator=rf_imp_para.estimators_[10]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = under_X_train.columns,
                class_names = True,
                rounded = True, proportion = False, 
                precision = 2, filled = True)
# Convert to png using system command (requires Graphviz)
import pydot
os.environ["PATH"] += os.pathsep + 'C:/Appl/release/bin'
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree_5.png')



#you can save the model for future use with the code below
import pickle

rfdump=rf_imp_para
filename = 'single_RFC_downsamp.joblib'
pickle.dump(rfdump, open(filename, 'wb'))

#_________________MEASURE PERFORMANCE



#______________ROC
#upsampled+important features
rfu_roc_auc = roc_auc_score(y_test, rf_imp_up.predict(imp_X_test))
rfu_fpr, rfu_tpr, rfu_thresholds = roc_curve(y_test, rf_imp_up.predict_proba(imp_X_test)[:,1])
#important features
rf_roc_auc = roc_auc_score(y_test, rf.predict(imp_X_test))
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(imp_X_test)[:,1])

plt.figure()
#plt.plot(rfu_fpr, rfu_tpr, label='Rand For Upsampled (area = %0.2f)' % rfu_roc_auc)
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC')
plt.show()

#________________PRECISION RECALL CURVE

# precision-recall curve and f1
#from sklearn.datasets import make_classification
#from sklearn.neighbors import KNeighborsClassifier

rf_imp=rf
# predict probabilities
probs = rf_imp.predict_proba(imp_X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = rf_imp.predict(imp_X_test)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)
# calculate F1 score
f1 = f1_score(y_test, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(y_test, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the roc curve for the model
plt.plot(recall, precision, marker='.')
# show the plot
plt.show()



#______________________FIND THE BEST PARAMETERS
rf = RandomForestClassifier(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 3)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 120, num = 3)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

# Use the random grid to search for best hyperparameters

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(under_X_train, under_y_train)

rf_random.best_params_

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
report(rf_random.cv_results_)

##_____grid search CV
#
#from sklearn.model_selection import GridSearchCV
## Create the parameter grid based on the results of random search 
#param_grid = {
#    'bootstrap': [True],
#    'max_depth': [20, 30, 40, 100],
#    'max_features': [5, 10],
#    'min_samples_leaf': [1, 2, 3],
#    'min_samples_split': [3, 5, 7],
#    'n_estimators': [1000, 1500, 2000, 2500]
#}
## Create a based model
#rf = RandomForestClassifier()
## Instantiate the grid search model
#grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
#
## Fit the grid search to the data
#grid_search.fit(X_train, y_train)
#grid_search.best_params_


#________________ARCHIVE


##_______________logistic regression
#
#
#logreg = LogisticRegression()
#logreg.fit(X_train, y_train)
#
#print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(y_test, logreg.predict(X_test))))


##_______________SVM
#
#from sklearn.svm import SVC
#svc = SVC()
#svc.fit(X_train, y_train)
#
#print('Support vector machine accuracy: {:.3f}'.format(accuracy_score(y_test, svc.predict(X_test))))


#__________cross validation

#from sklearn import model_selection
#from sklearn.model_selection import cross_val_score
#
#kfold = model_selection.KFold(n_splits=10, random_state=7)
#modelCV = RandomForestClassifier()
#scoring = 'accuracy'
#results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
#print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


#print(classification_report(y_test, logreg.predict(X_test)))
#log reg is useless for leavers

##Feature importance plot
#
## list of x locations for plotting
#x_values = list(range(len(importance)))
## Make a bar chart
#plt.bar(x_values, importance, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
## Tick labels for x axis
#plt.xticks(x_values, cols, rotation='vertical')
## Axis labels and title
#plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
#plt.show()



