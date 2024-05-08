import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
import multiprocessing  # Import the multiprocessing library

random.seed(100000)




# Define a function to process each scenario
def process_scenario(scenario_name):

    # Loading Data
    df = pd.read_csv(f"logs/ParaTunningFiles/OptimalCost2100/ParaTunning_OptimalCost_Gulf_2100_{scenario_name}.csv") 

    # Filter out the 'bootstrap' column
    df_filtered = df.drop(columns=['bootstrap'])

    # Reshape the data for creating a heatmap
    heatmap_data = df_filtered.pivot_table(index=['n_estimators'], columns=['max_depth'], values='mean_test_score')

    # Create a heatmap using seaborn with mean test values formatted to three digits
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='coolwarm', cbar_kws={'label': 'mean_test_score'})
    plt.title('Heatmap of mean_test_score with Categorical Parameters')
    plt.savefig(f'logs/ParaTunningPlots/Heatmaps/OptimalCost2100/SSPRCP{scenario_name}/OptimalCost_Gulf_2100_{scenario_name}_n_estimators_vs_max_depth.png')
    plt.close()
    
    # Reshape the data for creating a heatmap
    heatmap_data = df_filtered.pivot_table(index=['n_estimators'], columns=['min_samples_split'], values='mean_test_score')

    # Create a heatmap using seaborn with mean test values formatted to three digits
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='coolwarm', cbar_kws={'label': 'mean_test_score'})
    plt.title('Heatmap of mean_test_score with Categorical Parameters')
    plt.savefig(f'logs/ParaTunningPlots/Heatmaps/OptimalCost2100/SSPRCP{scenario_name}/OptimalCost_Gulf_2100_{scenario_name}_n_estimators_vs_min_samples_split.png')
    plt.close()
    
    
    # Reshape the data for creating a heatmap
    heatmap_data = df_filtered.pivot_table(index=['max_depth'], columns=['min_samples_split'], values='mean_test_score')

    # Create a heatmap using seaborn with mean test values formatted to three digits
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='coolwarm', cbar_kws={'label': 'mean_test_score'})
    plt.title('Heatmap of mean_test_score with Categorical Parameters')
    plt.savefig(f'logs/ParaTunningPlots/Heatmaps/OptimalCost2100/SSPRCP{scenario_name}/OptimalCost_Gulf_2100_{scenario_name}_max_depth_vs_min_samples_split.png')
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
