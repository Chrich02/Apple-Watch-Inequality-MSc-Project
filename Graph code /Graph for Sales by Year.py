import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower

# Statista data
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Sales in Millions': [8.5, 11.6, 17.7, 22.5, 28.4, 31.1, 39.4, 53.9, 38]
}
df = pd.DataFrame(data)

# --- Trend Statistics ---
total_sales = df['Sales in Millions'].sum()
average_sales = df['Sales in Millions'].mean()
max_sales = df['Sales in Millions'].max()
year_with_max = df[df['Sales in Millions'] == max_sales]['Year'].iloc[0]
min_sales = df['Sales in Millions'].min()
year_with_min = df[df['Sales in Millions'] == min_sales]['Year'].iloc[0]

# Calculate the percentage change year-over-year
df['Year-over-Year Change (%)'] = df['Sales in Millions'].pct_change() * 100
df.loc[0, 'Year-over-Year Change (%)'] = 0 # First year has no change

# --- Add Correlation and P-value Calculation ---
# The number of years is our sample size
n_years = len(df)

# Calculate Pearson's r and the p-value
correlation, p_value = stats.pearsonr(df['Year'], df['Sales in Millions'])


print("--- Trend Statistics ---")
print(f"Total sales from 2015-2023: {total_sales:.2f} million")
print(f"Average sales per year: {average_sales:.2f} million")
print(f"Year with the most sales: {year_with_max} with {max_sales} million")
print(f"Year with the fewest sales: {year_with_min} with {min_sales} million")
print("\nYear-over-Year Percentage Change in Sales:")
print(df[['Year', 'Year-over-Year Change (%)']].to_string(index=False))

# Add statistical reporting
print("\n--- Correlation Analysis ---")
print(f"Pearson's r: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")

# --- Plotting the graph ---
# Create a PDF file to save the plot
pdf_pages = PdfPages('Apple_Watch_Sales.pdf')

fig, ax = plt.subplots(figsize=(10, 6))

# Create the bar chart
bars = ax.bar(df['Year'], df['Sales in Millions'], color='skyblue', edgecolor='black')

# Add labels and a title
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Sales (in Millions)', fontsize=12)
ax.set_title('Apple Watch Sales by Year (2015-2023)', fontsize=14, pad=20)
ax.set_xticks(df['Year'])
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add the count on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.1f}',
            ha='center', va='bottom', fontsize=10)

# Calculate and plot the line of best fit (trend line)
z = np.polyfit(df['Year'], df['Sales in Millions'], 1)
p = np.poly1d(z)
ax.plot(df['Year'], p(df['Year']), "r--", label="Trend Line")
ax.legend()

# Save the plot to the PDF
pdf_pages.savefig(fig, bbox_inches='tight')

# Close the PDF file
pdf_pages.close()

print("\nPDF graph 'Apple_Watch_Sales.pdf' has been created successfully!")