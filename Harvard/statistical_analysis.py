import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Load the data
df = pd.read_csv('participant_summary_corrected.csv')

# Display the first few rows and information about the DataFrame
print(df.head())
print(df.info())

# 1. Average the HR across activities for each participant and device
df_avg_hr = df.groupby(['participant_id', 'gender', 'age', 'BMI', 'device'])['heart_rate'].mean().reset_index()

# 2. Pivot the table to have AW and FB heart rates in separate columns
df_pivoted = df_avg_hr.pivot_table(index=['participant_id', 'gender', 'age', 'BMI'], columns='device', values='heart_rate').reset_index()

# Rename the columns for clarity
df_pivoted.columns.name = None
df_pivoted = df_pivoted.rename(columns={'AW': 'heart_rate_AW', 'FB': 'heart_rate_FB'})

# 3. Calculate the difference in heart rate
df_pivoted['HR_diff'] = df_pivoted['heart_rate_AW'] - df_pivoted['heart_rate_FB']

# Save the processed DataFrame to a CSV file for the user
df_pivoted.to_csv('processed_participant_data.csv', index=False)
print("Processed data saved to processed_participant_data.csv")
print(df_pivoted.head())
print(df_pivoted.info())

# 4. Perform a paired t-test for the main effect of AW vs FB
t_stat, p_value_ttest = stats.ttest_rel(df_pivoted['heart_rate_AW'], df_pivoted['heart_rate_FB'])
print("\n--- Paired T-test (AW vs FB) ---")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value_ttest:.4f}")

# Calculate Cohen's d for paired t-test
mean_diff = df_pivoted['HR_diff'].mean()
std_diff = df_pivoted['HR_diff'].std()
cohen_d = mean_diff / std_diff
print(f"Cohen's d: {cohen_d:.4f}")

# 5. ANOVA
# For gender, it's a binary categorical variable
print("\n--- ANOVA: HR_diff vs Gender ---")
model_gender = ols('HR_diff ~ C(gender)', data=df_pivoted).fit()
anova_gender = anova_lm(model_gender, typ=2)
print(anova_gender)

# Calculate eta-squared for gender
ss_gender = anova_gender['sum_sq']['C(gender)']
ss_residual_gender = anova_gender['sum_sq']['Residual']
eta_squared_gender = ss_gender / (ss_gender + ss_residual_gender)
print(f"Eta-squared ($\eta^2$): {eta_squared_gender:.4f}")

# For BMI and age, convert the continuous data into categories for ANOVA
df_pivoted['BMI_group'] = pd.qcut(df_pivoted['BMI'], q=2, labels=['Low BMI', 'High BMI'])
df_pivoted['age_group'] = pd.qcut(df_pivoted['age'], q=2, labels=['Young', 'Old'])

# ANOVA for BMI group
print("\n--- ANOVA: HR_diff vs BMI Group ---")
model_bmi = ols('HR_diff ~ C(BMI_group)', data=df_pivoted).fit()
anova_bmi = anova_lm(model_bmi, typ=2)
print(anova_bmi)

# Calculate eta-squared for BMI group
ss_bmi = anova_bmi['sum_sq']['C(BMI_group)']
ss_residual_bmi = anova_bmi['sum_sq']['Residual']
eta_squared_bmi = ss_bmi / (ss_bmi + ss_residual_bmi)
print(f"Eta-squared ($\eta^2$): {eta_squared_bmi:.4f}")

# ANOVA for age group
print("\n--- ANOVA: HR_diff vs Age Group ---")
model_age = ols('HR_diff ~ C(age_group)', data=df_pivoted).fit()
anova_age = anova_lm(model_age, typ=2)
print(anova_age)

# Calculate eta-squared for age group
ss_age = anova_age['sum_sq']['C(age_group)']
ss_residual_age = anova_age['sum_sq']['Residual']
eta_squared_age = ss_age / (ss_age + ss_residual_age)
print(f"Eta-squared ($\eta^2$): {eta_squared_age:.4f}")