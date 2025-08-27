import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load the data from the CSV file
file_path = "/Users/chris/Documents/Keele/Year 4/Semester 3/Project (Dissertaion) /Datasets (not GitHub)/processed_participant_data.csv"
df = pd.read_csv(file_path)

# Perform and report the correlation
corr, p_value = pearsonr(df['heart_rate_AW'], df['heart_rate_FB'])

print(f"Pearson correlation (r) = {corr:.2f}")
print(f"p-value = {p_value:.3f}")

# Create the scatter plot with a regression line
sns.set_style("whitegrid")
plt.figure(figsize=(10, 8))

# Create the scatter plot with a regression line using seaborn's regplot
# The 'ci=95' adds a 95% confidence interval band around the regression line
# The x and y variables are the Apple Watch and Fitbit heart rates
plot = sns.regplot(x='heart_rate_FB', y='heart_rate_AW', data=df, ci=95,
                   line_kws={'color': 'red'}, scatter_kws={'alpha': 0.7})

# Add titles and labels for clarity
plt.title('Relationship Between Apple Watch and Fitbit Heart Rate Measurements', fontsize=16)
plt.xlabel('Average Fitbit Heart Rate (BPM)', fontsize=12)
plt.ylabel('Average Apple Watch Heart Rate (BPM)', fontsize=12)

# Set the limits to make the plot's scale clearer and symmetrical if needed
plt.xlim(0, 150)
plt.ylim(0, 150)

# Add a text box with the correlation and p-value for easy viewing
plt.text(120, 20, f'r = {corr:.2f}\np = {p_value:.3f}', fontsize=12,
         bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.9))

# Display the plot
plt.show()