import pandas as pd

# Step 1: Load both datasets
heart_df = pd.read_csv("healthcare_project/heart_dataset_jocelyndumlao_1000[1].csv")
health_df = pd.read_csv("healthcare_project/healthcare_dataset.csv")

# Step 2: Inspect for common columns
print("Heart Dataset Columns:", heart_df.columns)
print("Health Dataset Columns:", health_df.columns)

# Step 3: Reset index and align row counts
heart_df.reset_index(drop=True, inplace=True)
health_df.reset_index(drop=True, inplace=True)

# Trim both datasets to the same number of rows (smallest)
min_rows = min(len(heart_df), len(health_df))
heart_df = heart_df.iloc[:min_rows]
health_df = health_df.iloc[:min_rows]

# Step 4: Concatenate horizontally
merged_df = pd.concat([heart_df, health_df], axis=1)

# Step 5: Drop duplicate rows
merged_df.drop_duplicates(inplace=True)

# Step 6: Fill missing values
merged_df.fillna("Unknown", inplace=True)

# Step 7: Save merged file
merged_df.to_csv("healthcare_project/merged_clean_health_dataset.csv", index=False)
print("✅ Merged file saved at: healthcare_project/merged_clean_health_dataset.csv")
