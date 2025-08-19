import pandas as pd

#  Load dataset 
file_path = "/Users/chris/Documents/Keele/Year 4/Semester 3/Project (Dissertaion) /Data/Harvard/aw_fb_cleansed.csv"
df = pd.read_csv(file_path)

print("=== ORIGINAL DATA CHECK ===")
print("First few rows:")
print(df.head())
print(f"\nDataset shape: {df.shape}")

#  Create participant groups and assign IDs based on ORDER OF FIRST APPEARANCE 
# Get unique participant combinations in the order they first appear
unique_participants = (
    df.drop_duplicates(subset=['BMI', 'gender', 'age'])
    [['BMI', 'gender', 'age']]
    .reset_index(drop=True)
)

print(f"\n=== PARTICIPANT ORDER ===")
print("Participants in order of first appearance:")
print(unique_participants.head(10))

# Create participant_group mapping based on order of appearance
participant_mapping = unique_participants.copy()
participant_mapping['participant_group'] = range(1, len(participant_mapping) + 1)

# Merge this mapping back to the main dataframe
df = df.merge(participant_mapping, on=['BMI', 'gender', 'age'], how='left')

#  Assign participant IDs consistently across devices 
# Create ID mapping for both devices, maintaining the same numbering
id_map_list = []
for device in ['AW', 'FB']:
    device_map = participant_mapping.copy()
    device_map['device'] = device
    device_map['participant_id'] = device_map['participant_group']  # Same ID as group number
    id_map_list.append(device_map[['device', 'participant_group', 'participant_id']])

id_map = pd.concat(id_map_list, ignore_index=True)

# Merge back the participant IDs
df = df.merge(id_map[['device', 'participant_group', 'participant_id']], 
              on=['device', 'participant_group'], how='left')

# --- Verification ---
print(f"\n=== PARTICIPANT ID VERIFICATION ===")
verification = (
    df[['participant_id', 'BMI', 'gender', 'age', 'device']]
    .drop_duplicates()
    .sort_values(['participant_id', 'device'])
)
print("First 10 participant assignments:")
print(verification.head(20))

# Check specific participants
participant_1 = verification[verification['participant_id'] == 1].iloc[0]
participant_2 = verification[verification['participant_id'] == 2].iloc[0]

print(f"\nParticipant 1: Age {participant_1['age']}, BMI {participant_1['BMI']}, Gender {participant_1['gender']}")
print(f"Participant 2: Age {participant_2['age']}, BMI {participant_2['BMI']}, Gender {participant_2['gender']}")

#  Aggregate average HR per participant/activity 
print(f"\n=== AGGREGATING DATA ===")
summary_df = (
    df.groupby(['device', 'participant_id', 'gender', 'age', 'BMI', 'activity'], as_index=False)
    .agg(heart_rate=('heart_rate', 'mean'))
)

# --- Reorder columns ---
summary_df = summary_df[['participant_id', 'gender', 'age', 'BMI', 'heart_rate', 'device', 'activity']]

# --- Verification of final output ---
print(f"Summary dataframe shape: {summary_df.shape}")
print("First few rows of summary:")
print(summary_df.head(12))

# Check if BMI values are correct for first two participants
p1_rows = summary_df[summary_df['participant_id'] == 1]
p2_rows = summary_df[summary_df['participant_id'] == 2]

if len(p1_rows) > 0:
    print(f"\nParticipant 1 BMI in output: {p1_rows['BMI'].iloc[0]}")
    print(f"Participant 1 Age in output: {p1_rows['age'].iloc[0]}")

if len(p2_rows) > 0:
    print(f"Participant 2 BMI in output: {p2_rows['BMI'].iloc[0]}")
    print(f"Participant 2 Age in output: {p2_rows['age'].iloc[0]}")

# --- Save to CSV ---
output_path = "participant_summary_corrected.csv"
summary_df.to_csv(output_path, index=False)
print(f"\nAnalysis complete. Results saved to {output_path}")