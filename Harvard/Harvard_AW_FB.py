import pandas as pd

# --- Load dataset ---
file_path = "/Users/chris/Documents/Keele/Year 4/Semester 3/Project (Dissertaion) /Data/Harvard/aw_fb_clensed.csv"
df = pd.read_csv(file_path)

# --- Assign participant IDs ---
# Create unique participant groups by BMI + gender + age (to avoid clashes if BMI repeats)
df['participant_group'] = df.groupby(['BMI', 'gender', 'age']).ngroup() + 1

# Renumber IDs separately for AW and FB, but keep alignment across devices
id_map = (
    df[['device', 'participant_group']]
    .drop_duplicates()
    .sort_values(['participant_group', 'device'])
)

# Give consistent IDs starting at 1 per device
id_map['participant_id'] = id_map.groupby('device').cumcount() + 1

# Merge back IDs
df = df.merge(id_map, on=['device', 'participant_group'], how='left')

# --- Aggregate average HR per participant/activity ---
summary_df = (
    df.groupby(['device', 'participant_id', 'gender', 'age', 'BMI', 'activity'], as_index=False)
    .agg(heart_rate=('heart_rate', 'mean'))
)

# --- Reorder columns ---
summary_df = summary_df[['participant_id', 'gender', 'age', 'BMI', 'heart_rate', 'device', 'activity']]

# --- Save to CSV ---
output_path = "participant_summary.csv"
summary_df.to_csv(output_path, index=False)

print(f"✅ Analysis complete. Results saved to {output_path}")
