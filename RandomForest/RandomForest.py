### Set path ###
################

import os
os.chdir('C:/Users')


import numpy as np
import math
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import RandomForest_Methods as rfm
from rfpimp import *



### Data input ###
##################

data = pd.read_csv("HousePrices_Imputed.csv")

dataTarget = np.log(data[data.columns[len(data.columns)-1]])
dataExp = data[data.columns[0:len(data.columns)-1]]



### Preprocessing ###
#####################

# Get numeric and text features    
data_num = data[dataExp.select_dtypes(include=[np.number]).columns]
data_cat_pre = data[dataExp.select_dtypes(exclude=[np.number]).columns]

# Encode categorical variables 
enc = OneHotEncoder(handle_unknown='ignore')
enc_fit = enc.fit_transform(data_cat_pre[data_cat_pre.columns]).toarray()

enc_df = pd.DataFrame(enc_fit, columns = enc.get_feature_names(data_cat_pre.columns))
if len(data_num.columns) == 0:
    X = enc_df
else:
    X = data_num.join(enc_df)

# Split dataset in features and target variable
X_train, X_test, y_train, y_test = train_test_split(X, dataTarget, test_size=0.3, random_state=1) # 70% training and 30% test



### Supevised learning ###
##########################

# Create Decision Tree classifer object
OGF = RandomForestRegressor(criterion="mse")

# Train CART
OGF = OGF.fit(X_train,y_train)

# Evaluate CART
print("R2: {0:.3f}".format(OGF.score(X_test, y_test)))



### Unsupervised learning: Feature importance ###
#################################################

# Construct unsupervised model based on synthetic data
# Construct desired forest
syn = RandomForestClassifier(criterion="gini", oob_score=True, bootstrap=True)

USF = rfm.RandomForestUnsupervised(syn)
USF.make_forest(X_train)

# Evaluate the synthetic model
print("Score on synthetic: {0:.3f}".format(USF.Synthetic_Score))

# Calculate OOB importance measures
imp = oob_importances(USF.model, USF.Synthetic_X, USF.Synthetic_Y)

# Calculate drop column importance
# dimp = oob_dropcol_importances(SYN, synth[2].drop(['Synthetic'], axis=1), synth[2]["Synthetic"])


### To validate, compare results with using only the top 5 features
# # Get 5 most important features
# ImpFea = [imp.iloc[i].name for i in range(5)]

# # Create Random Forest for ImpFea
# RET = RandomForestRegressor(criterion='mse')

# # Train Random Forest for ImpFea
# RET = RET.fit(X_train[ImpFea], y_train)

# # Evaluate Random Forest for ImpFea
# print("R2: {0:.3f}".format(RET.score(X_test[ImpFea], y_test)))



### Unsupervised learning: Outlier detection ###
#################################################

# Create proximity matrix
Prox = rfm.proximityMatrix(USF.model, X_train)

# Obtain outlier measue
outlm = [sum(Prox[i]) for i in range(len(Prox))]

X_trainO = X_train.copy()
X_trainO["outl"] = outlm



### Unsupervised learning: Correlation heatmap ###
##################################################

# Create correlation heatmap
corM = feature_corr_matrix(X_train)

# # Create feature dependency heatmap
# featM = feature_dependence_matrix



### Unsupervised learning: Clustering ###
#########################################

# Create clustering
kmed = KMedoids(n_clusters = 2, metric = 'precomputed').fit(Prox)

# Evaluate clustering
print("Silhouette score: {0:.3f}".format(silhouette_score(X_train, kmed.labels_)))
print("Davies-Bouldin score: {0:.3f}".format(davies_bouldin_score(X_train, kmed.labels_)))






