
# Your Python code starts here

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random


from collections import OrderedDict
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

random.seed(100000)
# Loading Data
X1 = pd.read_csv("Data/parameters_subsample_sneasybrick.csv") 
X2 = pd.read_csv("Data/sneasybrick-ciam_CIAMparameterSamples_nyc_ssp1-26.csv") 
Y = pd.read_csv("Data/sneasybrick-ciam_NPVtrials_nyc_ssp1-26.csv") 
X = pd.concat([X1, X2], axis=1)
# Extracting Feature names
feature_names = X.columns
# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=10)
# Standardizing the data.
scaler = StandardScaler().fit(X_train[feature_names]) 

X_train[feature_names] = scaler.transform(X_train[feature_names])
X_test[feature_names] = scaler.transform(X_test[feature_names])
# Creating test and train sets
df_train = y_train.join(X_train)
df_test = y_test.join(X_test)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
# For SSP1-26
# Defining hyperparameters and fitting the model

reg=RandomForestRegressor(random_state=42)
param_grid = { 
            "n_estimators"      : [90, 100, 115, 125, 130, 133, 140, 150, 155, 160],
            "max_features"      : ["sqrt"],
            "min_samples_split" : [4],
            "max_depth": [27],
            "bootstrap": [True, False],
            }

CV5_reg = GridSearchCV(estimator=reg, param_grid=param_grid, cv= 5)
CV5_reg.fit(X_train, y_train)

# Getting the best parameters

CV5_reg.best_params_
# Defining hyperparameters and fitting the model

param_grid = { 
            "n_estimators"      : [140],
            "max_features"      : ["sqrt"],
            "min_samples_split" : [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            "max_depth": [27],
            "bootstrap": [True, False],
            }

CV5_reg = GridSearchCV(estimator=reg, param_grid=param_grid, cv= 5)
CV5_reg.fit(X_train, y_train)

# Getting the best parameters

CV5_reg.best_params_
# Defining hyperparameters and fitting the model

param_grid = { 
            "n_estimators"      : [140],
            "max_features"      : ["sqrt"],
            "min_samples_split" : [4],
            "max_depth": [10, 15, 18, 20, 25, 27, 30, 33, 37, 40],
            "bootstrap": [True, False],
            }

CV5_reg = GridSearchCV(estimator=reg, param_grid=param_grid, cv= 5)
CV5_reg.fit(X_train, y_train)

# Getting the best parameters

CV5_reg.best_params_
# Fitting the model with the best parameters

reg_cv5 = RandomForestRegressor(random_state=42, bootstrap=False, max_features='sqrt', min_samples_split = 4, 
n_estimators= 140, max_depth=27)
reg_cv5.fit(X_train, y_train)
pred=reg_cv5.predict(X_train)
pred_test=reg_cv5.predict(X_test)

# Calculating MSE

print("MSE for Random Forest on CV train data: ", "{:.10f}".format(float(mean_squared_error(y_train, pred))))
print("MSE for Random Forest on test data: ", "{:.10f}".format(float(mean_squared_error(y_test, pred_test))))


# Obtaining feature importance
feature_importance_cv5_ssp1_26 = reg_cv5.feature_importances_

# Sorting features according to importance
sorted_idx = np.argsort(feature_importance_cv5_ssp1_26)
pos = np.arange(sorted_idx.shape[0])

# Plotting feature importances
plt.barh(pos, feature_importance_cv5_ssp1_26[sorted_idx], align="center")

plt.yticks(pos, np.array(feature_names)[sorted_idx], size =5)

plt.title("Feature Importance")
plt.xlabel("Importance score")
plt.savefig('ssp1_26.png');


