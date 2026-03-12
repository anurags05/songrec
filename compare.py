import pandas as pd

file1 = "artists.csv"
file2 = "tracks_features.csv"

df1 = pd.read_csv(file1, nrows=1000)  # sample first 1000 rows
df2 = pd.read_csv(file2, nrows=1000)

# print("Columns in file1:", df1.columns.tolist())
# print("Columns in file2:", df2.columns.tolist())
print(df1.dtypes)
print(df2.dtypes)

# # Columns in file1 but not in file2
# print(set(df1.columns) - set(df2.columns))
# # Columns in file2 but not in file1
# print(set(df2.columns) - set(df1.columns))

chunk_size = 100000

for chunk1, chunk2 in zip(pd.read_csv(file1, chunksize=chunk_size),
                          pd.read_csv(file2, chunksize=chunk_size)):
    # Compare specific columns
    diff = chunk1['track_name'].equals(chunk2['track_name'])
    if not diff:
        print("Difference found in this chunk")