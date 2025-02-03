import pickle
import bz2
from collections import Counter
from Bio import SeqIO

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

def generate_kmers(sequence, k_min=3, k_max=10):
    """Generate k-mers of sizes k_min to k_max, avoiding k-mers with 'X'."""
    kmers = set()
    for k in range(k_min, k_max + 1):
        kmers.update(sequence[i:i + k] for i in range(len(sequence) - k + 1) if 'X' not in sequence[i:i + k])
    return kmers

def identify_family_and_classification(sequence, family_kmers_to_save):
    """Identify the family and classify as HOMO or NON-HOMO."""
    kmer_counts = Counter()
    homo_counts = Counter()
    non_homo_counts = Counter()

    # Generate k-mers
    kmers = generate_kmers(sequence)

    # Compare with known families
    for family, trained_family in family_kmers_to_save.items():
        matched_homo = kmers & trained_family.homo  # Intersection for fast lookup
        matched_non_homo = kmers & trained_family.non_homo
        total_matches = len(matched_homo) + len(matched_non_homo)

        if total_matches > 0:
            kmer_counts[family] = total_matches
            homo_counts[family] = len(matched_homo)
            non_homo_counts[family] = len(matched_non_homo)

    # Determine the most probable family
    predicted_family = kmer_counts.most_common(1)[0][0] if kmer_counts else "Unknown"

    # Determine classification
    if predicted_family != "Unknown" and predicted_family in homo_counts:
        if homo_counts[predicted_family] > non_homo_counts[predicted_family]:
            classification = "Yes, there is a chance of binding"
        else:
            classification = "No chance for binding"
    else:
        classification = "Unknown"

    return predicted_family, classification

def process_fasta(fasta_file, family_kmers_to_save):
    """Process a FASTA file and print the results."""
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence_id = record.id
        sequence = str(record.seq)

        predicted_family, classification = identify_family_and_classification(sequence, family_kmers_to_save)

        # Print the results
        print("-" * 80)
        print(f"Sequence ID: {sequence_id}")
        print(f"Predicted Family: {predicted_family}")
        print(f"Classification: {classification}")
        print("-" * 80)

# Main Execution
pickle_file = "family_kmers_to_save.pkl.bz2"
fasta_file = "bat_cov.fasta"

# Load trained k-mer data
family_kmers_to_save = load_kmers(pickle_file)

# Process the FASTA file and print results
process_fasta(fasta_file, family_kmers_to_save)
