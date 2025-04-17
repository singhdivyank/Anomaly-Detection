import pandas as pd

# Define input and output file paths
input_file = 'data/raw/smart_meter_data.csv'
output_file = 'data/raw/smart_meter_data_dtou.csv'

# Control variable to write header only once in the output CSV
first_chunk = True

# Process the large CSV file in chunks
for chunk in pd.read_csv(input_file, chunksize=10**6):
    # Filter out rows that have "Std" in the "stdorToU" column
    # Convert the column to string to ensure comparison works correctly
    filtered_chunk = chunk[chunk['stdorToU'].astype(str) != 'Std']
    
    # Append the filtered rows to the output file; write headers only in the first chunk
    if not first_chunk:
        filtered_chunk.to_csv(output_file, index=False, mode='a', header=False)
    else:
        filtered_chunk.to_csv(output_file, index=False, mode='w')
        first_chunk = False

print(f"Filtered CSV file saved to {output_file}")
