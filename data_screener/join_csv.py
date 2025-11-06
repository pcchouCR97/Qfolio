import pandas as pd

# --- A more direct way to merge your files ---

# 1. Load the CSV files.
#    index_col=0 tells pandas to use the first column as the index.
#    parse_dates=True tells pandas to understand that the index contains dates.
df1 = pd.read_csv('all_stocks_adjusted_close_52.csv', index_col=0, parse_dates=True)
df2 = pd.read_csv('all_stocks_adjusted_close_52_0.csv', index_col=0, parse_dates=True)

# 2. Merge the two DataFrames.
#    'combine_first' intelligently fills missing values, aligning on the Date index.
merged_df = df1.combine_first(df2)

# 3. Sort the index to make sure the dates are in chronological order.
merged_df.sort_index(inplace=True)

# 4. Save the merged DataFrame to a new CSV file.
#    We use index=True to make sure the 'Date' index is saved correctly.
merged_df.to_csv('all_stocks_adjusted_close_52_merged.csv', index=True)

print("Merge complete! 'all_stocks_adjusted_close_52_merged.csv' has been created correctly.")