# Required packages

import pandas as pd
from sklearn.model_selection import train_test_split

# @title Train test splitting
# Load dataset
try:
    df = pd.read_csv('./Animal_data.csv', encoding='latin-1')
except FileNotFoundError:
    print("The specified file was not found.")
    raise

print("Dataset loaded successfully...")

# Check the original dataset
print("Dataset size:", len(df))

print("Loading for filtration...")

# Verify required columns exist
required_columns = ['Host', 'Family']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' is missing in the dataset.")

# Filter out categories with fewer than 2 samples in 'Host_agg', and 'Family'
valid_hosts = df['Host'].value_counts()[df['Host'].value_counts() > 1].index
valid_families = df['Family'].value_counts()[df['Family'].value_counts() > 1].index

df_filtered = df[
    df['Host'].isin(valid_hosts) &
    df['Family'].isin(valid_families)
]

# Check the filtered dataset
print("Filtered dataset size:", len(df_filtered))

# Create a stratification column by combining 'Host_agg', 'Species_agg', and 'Family'
df_filtered = df_filtered.copy()  # Avoid SettingWithCopyWarning
df_filtered['Stratify_col'] = (
    df_filtered['Host'] + "_" +
    df_filtered['Family']
)

# Filter out classes in 'Stratify_col' with fewer than 2 samples
stratify_counts = df_filtered['Stratify_col'].value_counts()
valid_stratify_classes = stratify_counts[stratify_counts > 1].index
df_filtered = df_filtered[df_filtered['Stratify_col'].isin(valid_stratify_classes)]

# Check the filtered dataset size again
print("Filtered dataset size after removing single-sample stratify classes:", len(df_filtered))

# Split dataset
train_df, test_df = train_test_split(
    df_filtered,
    test_size=0.15,  # Adjust as needed
    stratify=df_filtered['Stratify_col'],  # Stratify by the chosen column
    random_state=42  # Ensure reproducibility
)
print("Stratification completed...")

print("Splitting test and train completed...")

# Verify the split
print("Train set size:", len(train_df))
print("Test set size:", len(test_df))
print("Train set distribution:")
print(train_df['Stratify_col'].value_counts())
print("Test set distribution:")
print(test_df['Stratify_col'].value_counts())
