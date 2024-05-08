import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import multiprocessing  # Import the multiprocessing library

from collections import OrderedDict
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
import traceback


random.seed(100000)

# Loading Data

X = pd.read_csv("Data/features.csv")

# Extracting Feature names
feature_names = X.columns

# Define the parameter grid with all the hyperparameters you want to tune
param_grid = { 
    "n_estimators": 420,
    "min_samples_split": 2,
    "max_depth": 35,
    "max_features": 37,
}


# Define a function to process each scenario
def process_scenario(year, scenario_name):

    # load the dataset as a pandas dataframe
    df = pd.read_csv(f"sneasy_files/NoAdaptCost_Gulf_{year}_Sneasy_{scenario_name}.csv") 

    # Calculate the average of each column using the mean() function

    column_averages = df.sum()

    # convert the series to a dataframe with a single column
    Y = pd.DataFrame(column_averages, columns=['AverageNoAdaptCost'])

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
    
    best_params = param_grid

    # Create the final model with the best parameters
    reg_cv5 = RandomForestRegressor(random_state=100, **best_params)
    reg_cv5.fit(X_train, y_train)

    # Predict on the training and test data
    pred = reg_cv5.predict(X_train)
    pred_test = reg_cv5.predict(X_test)

    # Obtaining feature importance
    feature_importance = reg_cv5.feature_importances_

    # Sorting features according to importance
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0])
    
    # Sort the feature importances and corresponding feature names
    sorted_features = np.array(feature_names)[sorted_idx][::-1]
    sorted_importances = feature_importance[sorted_idx][::-1]

    # Create a DataFrame containing the sorted feature names and importances
    importance_df = pd.DataFrame({'Feature': sorted_features, 'Importance': sorted_importances})
    
    folder_path = f"logs/FeatureImpFilesTimestamp/NoAdaptCost{year}/"
    print(folder_path)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the file path where you want to save the CSV file
    csv_file_path = f'{folder_path}/NoAdaptCost_Gulf_{year}_{scenario_name}.csv'

    # Save the DataFrame to a CSV file
    importance_df.to_csv(csv_file_path, index=False)



# Define a function to be executed in parallel
def parallel_execution(year, scenario_name):
    try:
        print(f"Processing year {year} scenario {scenario_name}...")
        process_scenario(year, scenario_name)
        print(f"Year {year} scenario {scenario_name} processing completed.")
    except Exception as e:
        print(f"Error processing year {year} scenario {scenario_name}: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    # Define the scenarios to be processed
    years = np.arange(2000, 2101, 10)
    scenarios = ["126", "245", "460", "585"]

    # Create a pool of worker processes
    num_tasks = 4  # Number of tasks to run in parallel
    pool = multiprocessing.Pool(processes=num_tasks)

    # Execute the scenarios for each year in parallel
    for year in years:
        print(year)
        for scenario in scenarios:
            pool.apply_async(parallel_execution, args=(year, scenario))

    # Close the pool
    pool.close()
    pool.join()

    # Print a completion message
    print("All scenarios processed.")
