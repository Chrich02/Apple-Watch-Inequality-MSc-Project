import pandas as pd

# Load the processed participant data
df_participants = pd.read_csv('processed_participant_data.csv')

# Calculate descriptive statistics for Age and BMI
age_bmi_stats = df_participants[['age', 'BMI']].agg(['mean', 'std'])
age_bmi_stats_rounded = age_bmi_stats.round(2)
print("--- Mean and Standard Deviation for Age and BMI ---")
print(age_bmi_stats_rounded)

# Calculate the counts for Gender
gender_counts = df_participants['gender'].value_counts()
print("\n Gender Counts")
print(gender_counts)
print(f"Total participants: {len(df_participants)}")

# Save the statistics to CSV files
age_bmi_stats_rounded.to_csv('age_bmi_descriptive_stats.csv')
gender_counts.to_csv('gender_counts.csv')
print("\nDescriptive statistics for Age and BMI saved to age_bmi_descriptive_stats.csv")
print("Gender counts saved to gender_counts.csv")