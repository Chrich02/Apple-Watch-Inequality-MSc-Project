import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower

# Data provided by the user
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    'Number of Studies': [2, 3, 6, 12, 5, 9, 18, 15, 14, 10, 14]
}
df = pd.DataFrame(data)

# --- Trend Statistics ---
total_studies = df['Number of Studies'].sum()
average_studies = df['Number of Studies'].mean()
max_studies = df['Number of Studies'].max()
year_with_max = df[df['Number of Studies'] == max_studies]['Year'].iloc[0]
min_studies = df['Number of Studies'].min()
year_with_min = df[df['Number of Studies'] == min_studies]['Year'].iloc[0]

# Calculate the percentage change year-over-year
df['Year-over-Year Change (%)'] = df['Number of Studies'].pct_change() * 100
df.loc[0, 'Year-over-Year Change (%)'] = 0 # First year has no change

# --- Add Correlation and P-value Calculation ---
# The number of years is our sample size
n_years = len(df)

# Calculate Pearson's r and the p-value
correlation, p_value = stats.pearsonr(df['Year'], df['Number of Studies'])

# --- Calculate Statistical Power ---
# Power analysis using the effect size (r) and sample size (n)
# Convert r to Cohen's f for power analysis
effect_size = correlation / np.sqrt(1 - correlation**2)
alpha = 0.05
power_analysis = TTestIndPower()
power = power_analysis.solve_power(
    effect_size=effect_size,
    nobs1=n_years,
    alpha=alpha,
    power=None,
    ratio=1,
    alternative='two-sided'
)

print("--- Trend Statistics ---")
print(f"Total studies from 2015-2025: {total_studies}")
print(f"Average studies per year: {average_studies:.2f}")
print(f"Year with the most studies: {year_with_max} with {max_studies} studies")
print(f"Year with the fewest studies: {year_with_min} with {min_studies} studies")
print("\nYear-over-Year Percentage Change in Studies:")
print(df[['Year', 'Year-over-Year Change (%)']].to_string(index=False))

# Add the new statistical reporting
print("\n--- Correlation and Power Analysis ---")
print(f"Pearson's r: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Statistical Power: {power:.4f}")

# --- Plotting the graph ---
# Create a PDF file to save the plot
pdf_pages = PdfPages('Apple_Watch_Clinical_Studies.pdf')

fig, ax = plt.subplots(figsize=(10, 6))

# Create the bar chart
bars = ax.bar(df['Year'], df['Number of Studies'], color='skyblue', edgecolor='black')

# Add labels and a title
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Number of Clinical Studies', fontsize=12)
ax.set_title('Instances of "Apple Watch" in Clinical Trials Database Since Release (2015-2025)', fontsize=14, pad=20)
ax.set_xticks(df['Year'])
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Ensure the y-axis only shows whole numbers
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Add the count on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height}',
            ha='center', va='bottom', fontsize=10)

# Calculate and plot the line of best fit (trend line)
z = np.polyfit(df['Year'], df['Number of Studies'], 1)
p = np.poly1d(z)
ax.plot(df['Year'], p(df['Year']), "r--", label="Trend Line")
ax.legend()

# Save the plot to the PDF
pdf_pages.savefig(fig, bbox_inches='tight')

# Close the PDF file
pdf_pages.close()

print("\nPDF graph 'Apple_Watch_Clinical_Studies.pdf' has been created successfully!")