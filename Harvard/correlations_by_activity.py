import pandas as pd
from scipy.stats import pearsonr
# Load the data from the CSV file
file_path = "/Users/chris/Documents/Keele/Year 4/Semester 3/Project (Dissertaion) /Datasets (not GitHub)/participant_summary_corrected.csv"
df = pd.read_csv(file_path)

# Get a list of all unique activities in the dataset
activities = df['activity'].unique()
print(f"Activities found: {activities}")
print("-" * 50)

# A dictionary to store the correlation results for each activity
correlation_results = {}

# Loop through each activity to calculate the correlation 
for activity in activities:
    # Filter the DataFrame to get only the current activity's data
    activity_df = df[df['activity'] == activity].copy()

    # Separate the data into Apple Watch and Fitbit groups
    aw_data = activity_df[activity_df['device'] == 'AW']['heart_rate']
    fb_data = activity_df[activity_df['device'] == 'FB']['heart_rate']

    # Ensure both lists have the same number of data points
    if len(aw_data) == len(fb_data):
        # Calculate the Pearson correlation coefficient and p-value
        r, p = pearsonr(aw_data, fb_data)
        correlation_results[activity] = {'r': r, 'p': p}

# Print the results in a structured format
print("Correlation Results by Activity:")
for activity, results in correlation_results.items():
    r = results['r']
    p = results['p']
    
    # Check for statistical significance
    significance = "Statistically Significant" if p < 0.05 else "Not Statistically Significant"
    
    print(f"Activity: {activity:<15} | r = {r:.2f} | p = {p:.3f} | {significance}")

