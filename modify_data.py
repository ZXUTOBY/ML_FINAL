import pandas as pd

# Load the dataset
df = pd.read_csv('train_data.csv')

# Drop unnecessary columns
df_cleaned = df.drop(columns=['link', 'source', 'site'])

# Save the cleaned version
df_cleaned.to_csv('data/train_data.csv', index=False)

print("Cleaned data saved to train_data_cleaned.csv")

