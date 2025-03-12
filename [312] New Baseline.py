#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
RANDOM_STATE = 42


# In[2]:

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
train = pd.concat(train_dfs, ignore_index=True)


# In[3]:


print(train.info())



# In[4]:


# List of columns to drop
drop_cols = [
    "timestampdt",
    "timestamp"
]

# Drop from trainml
trainml = train.drop(columns=drop_cols)
print(trainml.info())


# In[5]:


print(trainml.isna().sum().sort_values(ascending=False).head(20))



# In[6]:


trainbackup = trainml



# In[7]:



squarefeet = trainml["square_feet"]
target = np.log1p(trainml["meter_reading"]/squarefeet)
trainml = trainml.drop("meter_reading", axis = 1)



# In[8]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# 1) Identify all numeric columns in trainml
all_numeric_cols = trainml.select_dtypes(include=[np.number]).columns.tolist()

print("Numeric columns found:", all_numeric_cols)

# 2) Initialize the StandardScaler
scaler = StandardScaler()

# 3) Choose a suitable chunk size (depends on your available RAM)
chunk_size = 10_000

# ---------------------------------------------
# PASS 1: PARTIAL-FIT THE SCALER ON TRAIN IN CHUNKS
# ---------------------------------------------
train_nrows = trainml.shape[0]

for start in range(0, train_nrows, chunk_size):
    end = min(start + chunk_size, train_nrows)
    
    # Slice out a chunk
    chunk = trainml.iloc[start:end]
    
    # partial_fit updates running mean/variance for numeric columns
    scaler.partial_fit(chunk[all_numeric_cols])

print("Partial fit complete.")
print("Learned means:", scaler.mean_)
print("Learned variances:", scaler.var_)

# ---------------------------------------------
# PASS 2: TRANSFORM THE TRAIN SET IN CHUNKS
# ---------------------------------------------
transformed_train_chunks = []

for start in range(0, train_nrows, chunk_size):
    end = min(start + chunk_size, train_nrows)
    
    # Slice out the chunk
    chunk = trainml.iloc[start:end].copy()
    
    # Scale the numeric columns in this chunk
    chunk[all_numeric_cols] = scaler.transform(chunk[all_numeric_cols])
    
    # Collect transformed chunk
    transformed_train_chunks.append(chunk)

# Concatenate all transformed chunks
train_scaled = pd.concat(transformed_train_chunks, ignore_index=True)

print("Train scaling done. Final shape:", train_scaled.shape)


# ---------------------------------------------
# Now `train_scaled` and `test_scaled` contain 
# the scaled values for ALL numeric columns. 
# Any non-numeric columns remain unchanged.
# ---------------------------------------------


# In[10]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Features and target
X = train_scaled
y = target

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Defining the XGBoost regressor model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',  # For regression tasks
    n_estimators=100,              # Number of trees
    learning_rate=0.1,             # Learning rate
    max_depth=6,                   # Maximum tree depth
    random_state=RANDOM_STATE                # Seed for reproducibility
)

# Training the model
xgb_model.fit(X_train, y_train)

# 1) Import joblib and datetime
import joblib
from datetime import datetime

# 2) Create or use your model directory
model_dir = os.path.join(base_dir, "Models")  # /workspaces/ashare/Models
os.makedirs(model_dir, exist_ok=True)

# 3) Generate a date string (e.g., 20250314)
current_date = datetime.now().strftime('%Y%m%d')

# 4) Build a filename (e.g., xgb_model_20250314.joblib)
model_filename = f"xgb_model_{current_date}.joblib"

# 5) Full path to save the model
model_path = os.path.join(model_dir, model_filename)

# 6) Save the trained model
joblib.dump(xgb_model, model_path)

print(f"Model saved to: {model_path}")







