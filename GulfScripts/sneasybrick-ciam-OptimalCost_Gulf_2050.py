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
def process_scenario(scenario_name):

    # Loading Data

    # load the dataset as a pandas dataframe
    df = pd.read_csv(f"Data/OptimalCost_Gulf_2050_Sneasy_{scenario_name}.csv") 

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

    # Define the file path where you want to save the CSV file
    csv_file_path = f'logs/FeatureImpFiles/OptimalCost2050/OptimalCost_Gulf_2050_{scenario_name}.csv'

    # Save the DataFrame to a CSV file
    importance_df.to_csv(csv_file_path, index=False)

    # Plotting feature importances
    plt.barh(pos, feature_importance[sorted_idx], align="center")

    plt.yticks(pos, np.array(feature_names)[sorted_idx], size =5)

    plt.title(f"FeatureImp OptimalCost Gulf 2050 {scenario_name}")
    plt.xlabel("Importance score")
    plt.savefig(f'logs/FeatureImpPlots/OptimalCost2050/OptimalCost_Gulf_2050_{scenario_name}.png')
    plt.close()



# Define a function to be executed in parallel
def parallel_execution(scenario_name):
    print(f"Processing {scenario_name}...")
    process_scenario(scenario_name)
    print(f"{scenario_name} processing completed.")

if __name__ == '__main__':
    # Define the scenarios to be processed
    scenarios = ["126", "245", "460", "585"]

    # Create a pool of worker processes
    num_tasks = 4  # Number of tasks to run in parallel
    pool = multiprocessing.Pool(processes=num_tasks)

    # Execute the scenarios in parallel
    pool.map(parallel_execution, scenarios)

    # Close the pool
    pool.close()
    pool.join()

    # Print a completion message
    print("All scenarios processed.")
