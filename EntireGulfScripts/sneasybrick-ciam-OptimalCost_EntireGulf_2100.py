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

lonlat = pd.read_csv("Data/lonlat_gulf_list.csv") 
lonlat = lonlat.rename(columns={'x1': 'lon', 'x2': 'lat'})

# Create empty lists to collect feature importances and row numbers for each Y value

all_feature_importances = []
all_row_numbers = []

def process_scenario(scenario_name, para_grid):
    print(f"Processing {scenario_name} with parameters: {para_grid}...")

    # Clear the lists at the beginning of each scenario
    all_feature_importances.clear()
    all_row_numbers.clear()

    # load the dataset as a pandas dataframe
    df = pd.read_csv(f"Data/OptimalCost_Gulf_2100_Sneasy_{scenario_name}.csv") 

    best_params = para_grid
    print(best_params)


    # Loop through each row in the dataset as Y
    for row_index, row in df.iterrows():
        
        Y = row.to_frame()  # Convert the row to a DataFrame
        Y = Y.rename(columns={Y.columns[0]: 'AverageOptimalCost'}) 

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

        # Create the final model with the best parameters
        reg_cv5 = RandomForestRegressor(random_state=100, **best_params)
        reg_cv5.fit(X_train, y_train)

        # Calculate feature importance
        feature_importance = reg_cv5.feature_importances_

        # Sorting features according to importance
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0])

        
        # Append the feature importances and row number to the respective lists
        all_feature_importances.append(feature_importance)
        all_row_numbers.append(row_index)


    # Inside the process_scenario function, create a stacked plot for the two specific features
    specific_feature_names = ['movefactor_s', 'climate_sensitivity', 'antarctic_temp_threshold']

    # Find the indices of the specific features in feature_names
    specific_feature_indices = [feature_names.get_loc(feature) for feature in specific_feature_names]

    # Create a stacked plot for the two specific features
    fig, ax = plt.subplots(figsize=(10, 6))
    stacked_feature_importances = np.array(all_feature_importances[0:21])
    row_labels = [f"{row_num}" for row_num in all_row_numbers[0:21]]

    # Calculate the bottom position for each bar
    bottom = np.zeros(len(row_labels))

    for i, feature_index in enumerate(specific_feature_indices):
        feature_importance = stacked_feature_importances[:, feature_index]
        ax.bar(row_labels, feature_importance, label=specific_feature_names[i], bottom=bottom)
        bottom += feature_importance

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Features')
    plt.title(f"FeatureImp OptimalCost EntireGulf 2100 {scenario_name}")
    plt.xlabel("Segments")
    plt.ylabel("Feature Importance")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"logs/FeatureImpEntireGulfStackBarPlots/OptimalCost2100/FeatureImp_OptimalCost_EntireGulf_2100_{scenario_name}.png")

    # Create a DataFrame for the feature importances
    feature_importance_df = pd.DataFrame(all_feature_importances, columns=feature_names)  # Assuming feature_names has the names of all features

    # Concatenate the latitude and longitude DataFrame with the feature importance DataFrame
    merged_df = pd.concat([lonlat, feature_importance_df], axis=1)

    # Define the path where you want to save the merged DataFrame
    save_path = f"logs/FeatureImpEntireGulfFiles/OptimalCost2100/FeatureImp_OptimalCost_EntireGulf_2100_{scenario_name}.csv"

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(save_path, index=False)



# Define a function for parallel execution
def parallel_execution(scenario_name, para_grid):
    print(f"Processing {scenario_name}...")
    process_scenario(scenario_name, para_grid[scenario_name])
    print(f"{scenario_name} processing completed.")

if __name__ == '__main__':
    # Define the scenarios
    scenarios = ["126", "245", "460", "585"]

    # Define the parameter grids for each scenario
    para_grids = {
        "126": {"n_estimators": 420, "min_samples_split": 2, "max_depth": 35, "max_features": 37},
        "245": {"n_estimators": 420, "min_samples_split": 2, "max_depth": 35, "max_features": 37},
        "460": {"n_estimators": 420, "min_samples_split": 2, "max_depth": 35, "max_features": 37},
        "585": {"n_estimators": 420, "min_samples_split": 2, "max_depth": 35, "max_features": 37}
    }

    # Create a pool of worker processes
    num_tasks = 4  # Number of tasks to run in parallel
    pool = multiprocessing.Pool(processes=num_tasks)

    # Execute the scenarios in parallel
    for scenario in scenarios:
        pool.apply_async(parallel_execution, args=(scenario, para_grids))

    # Close the pool
    pool.close()
    pool.join()

    # Print a completion message
    print("All scenarios processed.")