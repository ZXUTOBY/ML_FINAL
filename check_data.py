import pandas as pd







#Write new file
input_path = "/Users/toby_xu/ML_FINAL/data/train_data.csv"  # Replace with the actual file name
output_path = "/Users/toby_xu/ML_FINAL/data/first_20_rows.csv"

# Write only the first 20 rows
df = pd.read_csv(input_path, nrows=20)
df.to_csv(output_path, index=False)
