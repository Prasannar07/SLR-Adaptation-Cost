import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
import multiprocessing  # Import the multiprocessing library

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
import pickle

random.seed(100000)

# Loading Data

X1 = pd.read_csv("Data/parameters_subsample_sneasybrick.csv") 
# Read CSV files
X2 = pd.read_csv("Data/dvbm_s.csv")
X3 = pd.read_csv("Data/movefactor_s.csv")
X4 = pd.read_csv("Data/vslel_s.csv")
X5 = pd.read_csv("Data/vslmult_s.csv")
X6 = pd.read_csv("Data/wvel_s.csv")
X7 = pd.read_csv("Data/wvpdl_s.csv")

# Rename the columns
X2.rename(columns={X2.columns[0]: 'dvbm_s'}, inplace=True)
X3.rename(columns={X3.columns[0]: 'movefactor_s'}, inplace=True)
X4.rename(columns={X4.columns[0]: 'vslel_s'}, inplace=True)
X5.rename(columns={X5.columns[0]: 'vslmult_s'}, inplace=True)
X6.rename(columns={X6.columns[0]: 'wvel_s'}, inplace=True)
X7.rename(columns={X7.columns[0]: 'wvpdl_s'}, inplace=True)

X = pd.concat([X1, X2, X3, X4, X5, X6, X7], axis=1)

# Extracting Feature names
feature_names = X.columns

# Define the parameter grid with all the hyperparameters you want to tune
param_grid = { 
    "n_estimators": [140, 150, 160, 170, 180, 200, 230, 255, 270, 300, 325, 350],
    "max_features": ["sqrt"],
    "min_samples_split": [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 40, 42],
    "max_depth": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    "bootstrap": [True, False],
}

# Create empty lists to collect data from all scenarios
all_top_feature_indices = []
all_top_feature_importances = []

# Define a function to process each scenario
def process_scenario(scenario_name):

   # Clear the lists at the beginning of each scenario
    all_top_feature_indices.clear()
    all_top_feature_importances.clear()

    # Loading Data

    # load the dataset as a pandas dataframe
    df = pd.read_csv(f"Data/OptimalCost_Gulf_2100_Sneasy_{scenario_name}.csv") 

    # Calculate the average of each column using the mean() function

    column_averages = df.sum()

    # convert the series to a dataframe with a single column
    Y = pd.DataFrame(column_averages, columns=['AverageOptimalCost'])

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=40)

    # Standardizing the data.
    scaler = StandardScaler().fit(X_train[feature_names]) 

    X_train[feature_names] = scaler.transform(X_train[feature_names])
    X_test[feature_names] = scaler.transform(X_test[feature_names])

    # Creating test and train sets
    df_train = y_train.join(X_train)
    df_test = y_test.join(X_test)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Create the RandomForestRegressor with your chosen random_state
    reg = RandomForestRegressor(random_state=100)

    # Perform the grid search with cross-validation
    CV5_reg = GridSearchCV(estimator=reg, param_grid=param_grid, cv=5)

    # Perform the grid search with cross-validation 
    CV5_reg.fit(X_train, y_train)

    # After fitting the GridSearchCV, you can access the results
    results = CV5_reg.cv_results_

    # Extract the mean test scores and parameters for all combinations
     
    mean_test_scores = results['mean_test_score']
    n_estimators = results['param_n_estimators']
    min_samples_split = results['param_min_samples_split']
    max_depth = results['param_max_depth']
    bootstrap = results['param_bootstrap']



    # Create a DataFrame
    result_df = pd.DataFrame({
        'mean_test_score': mean_test_scores,
        'n_estimators': n_estimators,
        'min_samples_split': min_samples_split,
        'max_depth': max_depth,
        'bootstrap': bootstrap,
    })

    # Sort the DataFrame by 'mean_test_score' in descending order
    result_df_sorted = result_df.sort_values(by='mean_test_score', ascending=False)
    
    # Save the DataFrame to a CSV file
    result_df_sorted.to_csv(f"ParaTunningFiles/OptimalCost2100/ParaTunning_OptimalCost_Gulf_2100_{scenario_name}.csv", index=False)

    # Getting the best parameters
    best_params = CV5_reg.best_params_
    print(f"Best Parameters for OptimalCost Gulf 2100 SSP_{scenario_name}:", best_params)
    
    return result_df_sorted

    
    
    
# Define a function to be executed in parallel
def parallel_execution(scenario_name):
    print(f"Processing {scenario_name}...")
    result_df = process_scenario(scenario_name)
    print(f"{scenario_name} processing completed.")
    return scenario_name, result_df

if __name__ == '__main__':
    # Define the scenarios to be processed
    scenarios = ["126", "245", "460", "585"]

    # Create a pool of worker processes
    num_tasks = 4  # Number of tasks to run in parallel
    pool = multiprocessing.Pool(processes=num_tasks)

    # Execute the scenarios in parallel
    results = pool.map(parallel_execution, scenarios)

    # Close the pool
    pool.close()
    pool.join()

    # Create a dictionary to store the dataframes for each scenario
    scenario_dataframes = {scenario_name: result_df for scenario_name, result_df in results}
    
    file_path = "ParaTunningDict/OptimalCost2100.pkl"


    # Save the dictionary of dataframes to a file
    with open(file_path, 'wb') as file:
        pickle.dump(scenario_dataframes, file)

    # Print a completion message
    print("All scenarios processed.")
    
    
