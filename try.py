import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

# 1) Define paths to your Parquet files (use relative or Linux paths in Codespaces)
base_dir = os.path.dirname(os.path.abspath(__file__))   # Directory where your script is located
train_path = os.path.join(base_dir, "Parquets", "updated_2_train.parquet")
test_path  = os.path.join(base_dir, "Parquets", "updated_2_test.parquet")

# 2) Read the train Parquet file in chunks
train_file = pq.ParquetFile(train_path)

# Decide how many rows you want in each chunk:
# Lower batch_size -> uses less memory but takes more iterations.
batch_size = 50_000

train_dfs = []
for batch in train_file.iter_batches(batch_size=batch_size):
    # Convert to Pandas DataFrame
    df_chunk = batch.to_pandas()
    # Perform any processing on this chunk if needed
    # e.g., partial fit a scaler or transform data
    train_dfs.append(df_chunk)

# 3) Concatenate all train chunks into a single DataFrame
train_data = pd.concat(train_dfs, ignore_index=True)
print("Train data shape:", train_data.shape)
print(train_data.info())
# 4) Read the test Parquet file in chunks
test_file = pq.ParquetFile(test_path)

test_dfs = []
for batch in test_file.iter_batches(batch_size=batch_size):
    df_chunk = batch.to_pandas()
    test_dfs.append(df_chunk)

test_data = pd.concat(test_dfs, ignore_index=True)
print("Test data shape:", test_data.shape)

# 5) Now you have train_data and test_data loaded
# Add your ML or analysis code here
print(train_data.info())