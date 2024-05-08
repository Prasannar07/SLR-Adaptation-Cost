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
    df = pd.read_csv(f"Data/NPVOptimal_Gulf_Sneasy_{scenario_name}.csv") 

    # Calculate the average of each column using the mean() function

    column_averages = df.sum()

    # convert the series to a dataframe with a single column
    Y = pd.DataFrame(column_averages, columns=['AverageNPVOptimalCost'])

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
    result_df_sorted.to_csv(f"logs/ParaTunningFiles/NPVOptimal/ParaTunning_NPVOptimal_Gulf_{scenario_name}.csv", index=False)

    # Getting the best parameters
    best_params = CV5_reg.best_params_
    print(f"Best Parameters for NPVOptimal Gulf SSP_{scenario_name}:", best_params)

    # Create the final model with the best parameters
    reg_cv5 = RandomForestRegressor(random_state=100, **best_params)
    reg_cv5.fit(X_train, y_train)

    # Predict on the training and test data
    pred = reg_cv5.predict(X_train)
    pred_test = reg_cv5.predict(X_test)

    # Calculating MSE

    print(f"NPVOptimal Gulf MSE for Random Forest on CV train data for ssp {scenario_name}: ", "{:.10f}".format(float(mean_squared_error(y_train, pred))))
    print(f"NPVOptimal Gulf MSE for Random Forest on test data for ssp {scenario_name}: ", "{:.10f}".format(float(mean_squared_error(y_test, pred_test))))


    # Obtaining feature importance
    feature_importance = reg_cv5.feature_importances_

    # Sorting features according to importance
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0])

    # Plotting feature importances
    plt.barh(pos, feature_importance[sorted_idx], align="center")

    plt.yticks(pos, np.array(feature_names)[sorted_idx], size =5)

    plt.title(f"FeatureImp NPVOptimal Gulf {scenario_name}")

    plt.xlabel("Importance score")
    plt.savefig(f'logs/FeatureImpPlots/NPVOptimal/NPVOptimal_Gulf_{scenario_name}.png')
    plt.close()

    # Append the top feature indices and importances to the all lists
    all_top_feature_indices.append(all_top_feature_indices)
    all_top_feature_importances.append(all_top_feature_importances)


    # Define range of n_estimators to test
    n_estimators_range = range(50, 1000, 100)

    # Get train and test scores for different values of n_estimators
    train_scores = []
    test_scores = []
    cv_scores = []
    for n in n_estimators_range:
        rf = RandomForestRegressor(n_estimators=n, random_state=100)
        rf.fit(X_train, y_train)
        train_scores.append(np.mean((rf.predict(X_train) - y_train) ** 2))
        test_scores.append(np.mean((rf.predict(X_test) - y_test) ** 2))
        cv_scores_n = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_scores.append(np.mean(-1 * cv_scores_n))

    train_scores1 = train_scores - np.min(train_scores)
    test_scores1 = test_scores - np.min(test_scores)
    cv_scores1 = cv_scores - np.min(cv_scores)

    # Calculate mean and standard deviation of train, test, and CV scores
    mean_train = np.mean(train_scores1)
    std_train = np.std(train_scores1, ddof=1)
    mean_test = np.mean(test_scores1)
    std_test = np.std(test_scores1, ddof=1)
    mean_cv = np.mean(cv_scores1)
    std_cv = np.std(cv_scores1, ddof=1)

    # Calculate 95% confidence intervals for train, test, and CV scores
    ci_train = 1.96 * std_train / np.sqrt(len(train_scores1))
    ci_test = 1.96 * std_test / np.sqrt(len(test_scores1))
    ci_cv = 1.96 * std_cv / np.sqrt(len(cv_scores1))

    # Plot train and test scores for different values of n_estimators
    plt.plot(n_estimators_range, train_scores1, label='Train')
    plt.fill_between(n_estimators_range, train_scores1 - ci_train, train_scores1 + ci_train, alpha=0.2)
    plt.plot(n_estimators_range, test_scores1, label='Test')
    plt.fill_between(n_estimators_range, test_scores1 - ci_test, test_scores1 + ci_test, alpha=0.2)
    plt.plot(n_estimators_range, cv_scores1, label='CV')
    plt.fill_between(n_estimators_range, cv_scores1 - ci_cv, cv_scores1 + ci_cv, alpha=0.2)
    plt.xlabel('n_estimators')
    plt.ylabel('MSE')
    plt.legend()
    plt.title(f"NPVOptimal Gulf {scenario_name} n estimator tuning")
    plt.savefig(f'logs/ParaTunningPlots/Linemaps/NPVOptimal/SSPRCP{scenario_name}/NPVOptimal_Gulf_{scenario_name}_n_estimator_tuning.png')
    plt.close()

    # Define range of n_estimators to test
    max_depth_range = range(15, 55, 5)

    # Get train and test scores for different values of n_estimators
    train_scores = []
    test_scores = []
    cv_scores = []
    for n in max_depth_range:
        rf = RandomForestRegressor(max_depth=n, random_state=100)
        rf.fit(X_train, y_train)
        train_scores.append(np.mean((rf.predict(X_train) - y_train) ** 2))
        test_scores.append(np.mean((rf.predict(X_test) - y_test) ** 2))
        cv_scores_n = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_scores.append(np.mean(-1 * cv_scores_n))

    train_scores1 = train_scores - np.min(train_scores)
    test_scores1 = test_scores - np.min(test_scores)
    cv_scores1 = cv_scores - np.min(cv_scores)

    # Calculate mean and standard deviation of train, test, and CV scores
    mean_train = np.mean(train_scores1)
    std_train = np.std(train_scores1, ddof=1)
    mean_test = np.mean(test_scores1)
    std_test = np.std(test_scores1, ddof=1)
    mean_cv = np.mean(cv_scores1)
    std_cv = np.std(cv_scores1, ddof=1)

    # Calculate 95% confidence intervals for train, test, and CV scores
    ci_train = 1.96 * std_train / np.sqrt(len(train_scores1))
    ci_test = 1.96 * std_test / np.sqrt(len(test_scores1))
    ci_cv = 1.96 * std_cv / np.sqrt(len(cv_scores1))

    # Plot train and test scores for different values of n_estimators
    plt.plot(max_depth_range, train_scores1, label='Train')
    plt.fill_between(max_depth_range, train_scores1 - ci_train, train_scores1 + ci_train, alpha=0.2)
    plt.plot(max_depth_range, test_scores1, label='Test')
    plt.fill_between(max_depth_range, test_scores1 - ci_test, test_scores1 + ci_test, alpha=0.2)
    plt.plot(max_depth_range, cv_scores1, label='CV')
    plt.fill_between(max_depth_range, cv_scores1 - ci_cv, cv_scores1 + ci_cv, alpha=0.2)
    plt.xlabel('max_depth')
    plt.ylabel('MSE')
    plt.legend()
    plt.title(f"NPVOptimal Gulf {scenario_name} max depth tuning")
    plt.savefig(f'logs/ParaTunningPlots/Linemaps/NPVOptimal/SSPRCP{scenario_name}/NPVOptimal_Gulf_{scenario_name}_max_depth_tuning.png')
    plt.close()


    # Define range of n_estimators to test
    min_samples_split_range = range(2, 20, 2)

    # Get train and test scores for different values of n_estimators
    train_scores = []
    test_scores = []
    cv_scores = []
    for n in min_samples_split_range:
        rf = RandomForestRegressor(min_samples_split=n, random_state=100)
        rf.fit(X_train, y_train)
        train_scores.append(np.mean((rf.predict(X_train) - y_train) ** 2))
        test_scores.append(np.mean((rf.predict(X_test) - y_test) ** 2))
        cv_scores_n = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_scores.append(np.mean(-1 * cv_scores_n))

    train_scores1 = train_scores - np.min(train_scores)
    test_scores1 = test_scores - np.min(test_scores)
    cv_scores1 = cv_scores - np.min(cv_scores)

    # Calculate mean and standard deviation of train, test, and CV scores
    mean_train = np.mean(train_scores1)
    std_train = np.std(train_scores1, ddof=1)
    mean_test = np.mean(test_scores1)
    std_test = np.std(test_scores1, ddof=1)
    mean_cv = np.mean(cv_scores1)
    std_cv = np.std(cv_scores1, ddof=1)

    # Calculate 95% confidence intervals for train, test, and CV scores
    ci_train = 1.96 * std_train / np.sqrt(len(train_scores1))
    ci_test = 1.96 * std_test / np.sqrt(len(test_scores1))
    ci_cv = 1.96 * std_cv / np.sqrt(len(cv_scores1))

    # Plot train and test scores for different values of n_estimators
    plt.plot(min_samples_split_range, train_scores1, label='Train')
    plt.fill_between(min_samples_split_range, train_scores1 - ci_train, train_scores1 + ci_train, alpha=0.2)
    plt.plot(min_samples_split_range, test_scores1, label='Test')
    plt.fill_between(min_samples_split_range, test_scores1 - ci_test, test_scores1 + ci_test, alpha=0.2)
    plt.plot(min_samples_split_range, cv_scores1, label='CV')
    plt.fill_between(min_samples_split_range, cv_scores1 - ci_cv, cv_scores1 + ci_cv, alpha=0.2)
    plt.xlabel('min_samples_split')
    plt.ylabel('MSE')
    plt.legend()
    plt.title(f"NPVOptimal Gulf {scenario_name} min samples split tuning")
    plt.savefig(f'logs/ParaTunningPlots/Linemaps/NPVOptimal/SSPRCP{scenario_name}/NPVOptimal_Gulf_{scenario_name}_min_samples_split_tuning.png')
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
