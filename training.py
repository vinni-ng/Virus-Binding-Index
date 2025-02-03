# Required packages
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import bz2file as bz2

train_df = pd.read_csv("./train_data.csv")

# Modify the defaultdict setup to avoid using lambda inside it
class FamilyKmers:
    def __init__(self):
        self.homo = defaultdict(set)  # Store unique k-mers as sets
        self.non_homo = defaultdict(set)  # Store unique k-mers as sets

def generate_kmers(sequence, k):
    """Generate overlapping k-mers from a sequence."""
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def filter_kmers(kmers):
    """Filter k-mers by excluding those containing 'X'."""
    return [kmer for kmer in kmers if 'X' not in kmer]

def filter_frequent_kmers(kmer_counts, min_occurrence=2):
    """
    Filter k-mers based on their occurrence across sequences.

    Args:
        kmer_counts (dict): A dictionary with k-mers as keys and their counts across sequences as values.
        min_occurrence (int): The minimum number of sequences in which a k-mer must appear to be included.

    Returns:
        list: Filtered list of k-mers that appear in more than the specified minimum number of sequences.
    """
    return [kmer for kmer, count in kmer_counts.items() if count >= min_occurrence]


def process_kmers_by_family(dataframe, k_range):
    """
    Process the dataframe and generate k-mers for each family,
    separating homo (human) and non-homo (non-human) k-mers.
    Removes k-mers that are found in both categories and filters k-mers
    that appear in more than one sequence.
    """
    family_kmers = defaultdict(FamilyKmers)
    kmer_counts = defaultdict(lambda: defaultdict(int))  # Track k-mer occurrences across sequences

    print("Started processing k-mers...")

    # Iterate through each row and extract sequences for homo and non-homo
    for _, row in dataframe.iterrows():
        sequence = row['Sequence']
        family = row['Family']
        human = row['Human']
        # Extract k-mers for the given range
        for k in k_range:
            kmers = generate_kmers(sequence, k)
            kmers = filter_kmers(kmers)  # Exclude k-mers containing 'X'

            # Count k-mer occurrences across sequences
            for kmer in kmers:
                kmer_counts[family][kmer] += 1

            # Separate into homo and non-homo
            if human == 1:  # Homo (human)
                for kmer in kmers:
                    family_kmers[family].homo[kmer] = 1
            else:  # Non-homo (non-human)
                for kmer in kmers:
                    family_kmers[family].non_homo[kmer] = 1

    print("Finished generating k-mers. Removing overlapping and infrequent k-mers...")

    # Remove overlapping k-mers and filter based on frequency
    for family, kmers_obj in family_kmers.items():
        homo_kmers_set = set(kmers_obj.homo.keys())
        non_homo_kmers_set = set(kmers_obj.non_homo.keys())

        # Find overlapping k-mers
        overlapping_kmers = homo_kmers_set & non_homo_kmers_set

        # Remove overlapping k-mers from both categories
        for kmer in overlapping_kmers:
            del kmers_obj.homo[kmer]
            del kmers_obj.non_homo[kmer]

        # Filter k-mers that appear in more than one sequence
        filtered_homo = filter_frequent_kmers(kmer_counts[family])
        filtered_non_homo = filter_frequent_kmers(kmer_counts[family])

        # Retain only the filtered k-mers in each category
        kmers_obj.homo = {kmer: 1 for kmer in filtered_homo if kmer in kmers_obj.homo}
        kmers_obj.non_homo = {kmer: 1 for kmer in filtered_non_homo if kmer in kmers_obj.non_homo}

    print("Overlapping and infrequent k-mers removed.")
    return family_kmers


def plot_kmers_distribution(family_kmers, k_range):
    """
    Plot the distribution of unique k-mers for each family.
    The plot will show the number of unique homo and non-homo k-mers for each family.
    Additionally, it will print the k-mer size with the maximum count for each family.
    """
    for family, kmers_obj in family_kmers.items():
        homo_counts = []
        non_homo_counts = []

        # For tracking unique k-mers for each size
        unique_homo_kmers = defaultdict(set)
        unique_non_homo_kmers = defaultdict(set)

        # Count unique k-mers for each size in the specified range
        for k in k_range:
            # Get the unique k-mers for size k
            homo_kmers = [kmer for kmer in kmers_obj.homo if len(kmer) == k]
            non_homo_kmers = [kmer for kmer in kmers_obj.non_homo if len(kmer) == k]

            # Add k-mers to the unique sets
            unique_homo_kmers[k] = set(homo_kmers)
            unique_non_homo_kmers[k] = set(non_homo_kmers)

            # Get the count of unique k-mers for size k
            homo_count = len(unique_homo_kmers[k])
            non_homo_count = len(unique_non_homo_kmers[k])

            homo_counts.append(homo_count)
            non_homo_counts.append(non_homo_count)

            # Print the number of unique k-mers for each size
            print(f"For family {family}, Homo (human) k-mers of size {k}: {homo_count} unique k-mers")
            print(f"For family {family}, Non-Homo (non-human) k-mers of size {k}: {non_homo_count} unique k-mers")

        # Determine the k-mer size with the maximum count for Homo and Non-Homo
        max_homo_k = k_range[homo_counts.index(max(homo_counts))] if homo_counts else None
        max_non_homo_k = k_range[non_homo_counts.index(max(non_homo_counts))] if non_homo_counts else None

        print(f"\nFor family {family}, the k-mer size with the max count for Homo (human): {max_homo_k} with {max(homo_counts)} unique k-mers")
        print(f"For family {family}, the k-mer size with the max count for Non-Homo (non-human): {max_non_homo_k} with {max(non_homo_counts)} unique k-mers\n")

        # Create the plot for unique k-mers
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, homo_counts, label=f'HOMO ({family})', color='blue', marker='o', linestyle='-', alpha=0.7)
        plt.plot(k_range, non_homo_counts, label=f'non-HOMO ({family})', color='orange', marker='o', linestyle='-', alpha=0.7)

        # Adding labels and title
        plt.title(f'Unique K-mer Distribution for {family} Family')
        plt.xlabel('K-mer Size')
        plt.ylabel('Unique K-mer Count')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

def save_kmers_up_to_max_count(family_kmers, k_range):
    """
    Save only up to the k-mer size where the maximum count was found for each family.
    """
    family_kmers_to_save = defaultdict(FamilyKmers)

    # Save only up to the k-mer size where the maximum count was found for each family
    for family, kmers_obj in family_kmers.items():
        # For Homo (human)
        homo_counts = [len([kmer for kmer in kmers_obj.homo if len(kmer) == k]) for k in k_range]
        max_homo_k = k_range[homo_counts.index(max(homo_counts))] if homo_counts else None

        # Save k-mers up to the max observed count for Homo (human)
        if max_homo_k:
            for k in range(3, max_homo_k + 1):
                for kmer in kmers_obj.homo:
                    if len(kmer) == k:
                        family_kmers_to_save[family].homo[kmer] = 1

        # For Non-Homo (non-human)
        non_homo_counts = [len([kmer for kmer in kmers_obj.non_homo if len(kmer) == k]) for k in k_range]
        max_non_homo_k = k_range[non_homo_counts.index(max(non_homo_counts))] if non_homo_counts else None

        # Save k-mers up to the max observed count for Non-Homo (non-human)
        if max_non_homo_k:
            for k in range(3, max_non_homo_k + 1):
                for kmer in kmers_obj.non_homo:
                    if len(kmer) == k:
                        family_kmers_to_save[family].non_homo[kmer] = 1

    # Save the family_kmers_to_save dictionary to a pickle file
    with open('family_kmers_to_save.pkl', 'wb') as f:
        pickle.dump(family_kmers_to_save, f)

    print("K-mers up to max count size have been saved to 'family_kmers_to_save.pkl'")


# Example usage:
# Assuming you have a dataframe `train_df` with the required columns: 'Sequence', 'Human', and 'Family'
# Define the k-mer size range
k_range = range(3, 10)  # From size 3 to 10

# Process k-mers by family
family_kmers = process_kmers_by_family(train_df, k_range)

# Plot the unique k-mer distributions for each family
plot_kmers_distribution(family_kmers, k_range)

# Save the k-mers up to the maximum count size
save_kmers_up_to_max_count(family_kmers, k_range)

# Optionally, save the family_kmers dictionary to a pickle file
with open('family_kmers_to_save.pkl', 'wb') as f:
    pickle.dump(family_kmers, f)

print("Unique K-mers up to max count size have been saved to 'family_kmers_to_save.pkl'")


def save_compressed_pickle(data, filename):
    """Save data to a compressed bz2 pickle file."""
    with bz2.BZ2File(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Compressed pickle file saved as {filename}")

# Save and compress the family_kmers dictionary
save_compressed_pickle(family_kmers, "family_kmers_to_save.pkl.bz2")

