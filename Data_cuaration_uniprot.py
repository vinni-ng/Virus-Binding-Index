# Required Packages
import pandas as pd
from Bio import SeqIO

# @title Data curation using Uniprot
# Function 1: Load metadata from an Excel file
def load_metadata(excel_file):
    """
    Load metadata from an Excel file into a DataFrame.
    """
    return pd.read_excel(excel_file)

# Function 2: Load sequences from a FASTA file
def load_sequences(fasta_file):
    """
    Load sequences from a FASTA file into a DataFrame.
    Extract relevant metadata from headers.
    """
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    sequence_data = []
    for record in sequences:
        # Extract primary ID and additional metadata
        header = record.description  # Full header
        primary_id = header.split('|')[1] if '|' in header else record.id  # Extract ID (e.g., P07612)
        protein_name = header.split(' ')[1] if ' ' in header else "Unknown"  # Extract protein name
        organism = None
        if "OS=" in header:
            organism = header.split("OS=")[1].split("OX=")[0].strip()  # Extract organism name

        # Append parsed data
        sequence_data.append({
            "ID": primary_id,
            "Protein": protein_name,
            "Organism": organism,
            "Sequence": str(record.seq),
            "Length": len(record.seq)
        })

    return pd.DataFrame(sequence_data)

# Function 3: Merge metadata and sequence data
def merge_data(metadata_df, sequences_df):
    """
    Merge metadata and sequence data on Entry and ID.
    """
    merged_df = pd.merge(metadata_df, sequences_df, how="inner", left_on="Entry", right_on="ID")
    return merged_df

# Main script
if __name__ == "__main__":
    # File paths
    excel_file = "./sequences_uniprot.xlsx"  # Path to the Excel file
    fasta_file = "./sequences_uniprot.fasta"  # Path to the FASTA file

    # Load metadata and sequences
    metadata_df = load_metadata(excel_file)
    sequences_df = load_sequences(fasta_file)

    # Merge data
    merged_df = merge_data(metadata_df, sequences_df)

    # Save the merged DataFrame to a CSV file
    output_file = "./curated_data_uniprot.csv"
    merged_df.to_csv(output_file, index=False)

    print(f"Merged data saved to {output_file}")
