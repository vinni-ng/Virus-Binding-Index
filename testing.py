# Required packages
import pickle
import bz2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# Load compressed pickle file
def load_kmers(pickle_file):
    """Load k-mers from a compressed pickle file."""
    with bz2.BZ2File(pickle_file, "rb") as file:
        family_kmers_to_save = pickle.load(file)

    # Convert k-mer dictionaries to sets for fast lookup
    for family, trained_family in family_kmers_to_save.items():
        trained_family.homo = set(trained_family.homo.keys())
        trained_family.non_homo = set(trained_family.non_homo.keys())

    return family_kmers_to_save


# Now you can call load_kmers function safely
pickle_file = "family_kmers_to_save.pkl.bz2"
family_kmers_to_save = load_kmers(pickle_file)

test_df = pd.read_csv("./test_data.csv")

def generate_kmers(sequence, k):
    """Generate k-mers of length k from a given sequence."""
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1) if 'X' not in sequence[i:i + k]]

def count_kmers(test_df, family_kmers_to_save):
    """Count filtered k-mers, HOMO and NON-HOMO k-mers for the correct family and add respective columns."""
    filtered_kmer_counts = []
    homo_counts = []
    non_homo_counts = []

    for _, row in test_df.iterrows():
        family = row['Family']
        sequence = row['Sequence']

        if family not in family_kmers_to_save:
            filtered_kmer_counts.append(0)
            homo_counts.append(0)
            non_homo_counts.append(0)
            continue

        trained_family = family_kmers_to_save[family]

        # Generate k-mers for lengths from 3 to 10
        filtered_kmers = [kmer for k in range(3, 11) for kmer in generate_kmers(sequence, k)]
        filtered_kmer_counts.append(len(filtered_kmers))

        # Count how many k-mers match the "homo" and "non-homo" sets
        homo_count = sum(1 for kmer in filtered_kmers if kmer in trained_family.homo)
        non_homo_count = sum(1 for kmer in filtered_kmers if kmer in trained_family.non_homo)

        homo_counts.append(homo_count)
        non_homo_counts.append(non_homo_count)

    # Add the counts to the DataFrame
    test_df['Filtered_Kmer_Count'] = filtered_kmer_counts
    test_df['HOMO_Count'] = homo_counts
    test_df['NON_HOMO_Count'] = non_homo_counts

    return test_df

def add_binds_column(test_df):
    """Add Binds column: 1 if HOMO_Count > NON_HOMO_Count, else 0."""
    test_df['Binds'] = (test_df['HOMO_Count'] > test_df['NON_HOMO_Count']).astype(int)
    return test_df

# Process the test data and count k-mers
test_df = count_kmers(test_df, family_kmers_to_save)

# Add 'Binds' column
test_df = add_binds_column(test_df)

# Clean the dataframe by keeping only relevant columns
columns_to_keep = ['Accession', 'Species', 'Family', 'Country', 'Host', 'Sequence', 'Virus',
                  'Human', 'HOMO_Count', 'NON_HOMO_Count', 'Binds']

test_df_cleaned = test_df[columns_to_keep]

# Save the cleaned dataframe to a new CSV file
test_df_cleaned.to_csv("results.csv", index=False)
print("Cleaned test dataframe saved to 'results.csv'")

# @title ROC_AUC

# Assuming `test_df` contains the 'Human' and 'Binds' columns
y_true = test_df['Human']  # True labels
y_pred = test_df['Binds']  # Predicted labels

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_true, y_pred)
print(f"ROC AUC Score: {roc_auc}")

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# @title ROC_AUC for each family
# Dictionary to store AUC scores for each family
family_auc_scores = {}

# Iterate over each family in the dataset
for family, df in test_df.groupby("Family"):
    if len(df["Plant"].unique()) < 2:  # Skip families that don't have both 0 and 1
        print(f"Skipping Family '{family}' (only one class present)")
        continue

    y_true = df["Human"]  # True labels
    y_pred = df["Binds"]  # Predicted labels

    # Compute ROC AUC score
    roc_auc = roc_auc_score(y_true, y_pred)
    family_auc_scores[family] = roc_auc

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # Plot ROC curve for each family
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {family}')
    plt.legend(loc="lower right")
    plt.show()

# Print all AUC scores
print("\nROC AUC Scores for Each Family:")
for family, auc in family_auc_scores.items():
    print(f"Family: {family}, AUC: {auc:.2f}")

# @title Checking false positives and false negatives
def calculate_false_positives_and_false_negatives(test_df):
    """
    Calculate the number of False Positives (FP) and False Negatives (FN) based on the 'Human' and 'Binds' columns.
    Also, return the accession IDs for False Positives.
    """
    false_positives = 0
    false_negatives = 0
    false_positive_accessions = []

    # Iterate through each row of the test dataframe
    for idx, row in test_df.iterrows():
        true_label = row['Human']
        predicted_label = row['Binds']
        accession_id = row.get('Accession', None)  # Assuming there's an 'Accession_ID' column

        # False Positive: Predicted 1 (human-like), but actual label is 0 (non-human)
        if predicted_label == 1 and true_label == 0:
            false_positives += 1
            if accession_id:
                false_positive_accessions.append(accession_id)

        # False Negative: Predicted 0 (non-human-like), but actual label is 1 (human)
        if predicted_label == 0 and true_label == 1:
            false_negatives += 1

    return false_positives, false_negatives, false_positive_accessions

# Example usage:
false_positives, false_negatives, false_positive_accessions = calculate_false_positives_and_false_negatives(test_df)

print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")
print(f"False Positive Accession IDs: {false_positive_accessions}")

