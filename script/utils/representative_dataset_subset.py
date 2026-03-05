import pandas as pd
import os

# Load the CSV file
input_csv = 'MLCQCodeSmellSamples.csv'
output_csv = 'RepresentativeSubset.csv'

# Read the CSV file into a DataFrame
try:
    df = pd.read_csv(input_csv, sep=';')
except FileNotFoundError:
    print(f"Error: The file {input_csv} does not exist.")
    exit(1)

# Group by 'smell' and take 50% from each category
smell_categories = df['smell'].unique()  # Get unique smell categories
representative_subset = pd.DataFrame()  # Initialize an empty DataFrame

for smell in smell_categories:
    subset = df[df['smell'] == smell]  # Filter for the current smell category
    sample_size = min(len(subset), int(0.2 * len(subset)))  # Calculate sample size (20% or total if less than 2)
    representative_subset = pd.concat([representative_subset, subset.sample(n=sample_size, random_state=1)])  # Append sampled data

# Save the representative subset to a new CSV file
representative_subset.to_csv(output_csv, index=False, sep=';')
print(f'Representative subset saved to {output_csv}')
