#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


train = pd.read_parquet(r'C:\Users\almon\Documents\Learning\Machine Learning\Kaggle\ASHARE\Cleaned\Parquet\updated_2_train.parquet')
test = pd.read_parquet(r'C:\Users\almon\Documents\Learning\Machine Learning\Kaggle\ASHARE\Cleaned\Parquet\updated_2_test.parquet')


# In[3]:


print(train.info())
print(test.info())


# In[4]:


# List of columns to drop
drop_cols = [
    "timestampdt",
    "timestamp"
]

# Drop from trainml
trainml = train.drop(columns=drop_cols)
testml = test.drop(columns=drop_cols)


# In[5]:


print(trainml.isna().sum().sort_values(ascending=False).head(20))
print(testml.isna().sum().sort_values(ascending=False).head(20))


# In[6]:


trainbackup = trainml
testbackup = testml


# In[7]:


row_id = testml["row_id"]
squarefeet = trainml["square_feet"]
target = np.log1p(trainml["meter_reading"]/squarefeet)
trainml = trainml.drop("meter_reading", axis = 1)
testml = testml.drop("row_id", axis = 1)


# In[8]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Suppose trainml and testml are your large DataFrames already in memory
# e.g.:
# trainml = pd.read_parquet("path/to/updated_train.parquet")
# testml  = pd.read_parquet("path/to/updated_test.parquet")

# 1) Identify all numeric columns in trainml
all_numeric_cols = trainml.select_dtypes(include=[np.number]).columns.tolist()

print("Numeric columns found:", all_numeric_cols)

# 2) Initialize the StandardScaler
scaler = StandardScaler()

# 3) Choose a suitable chunk size (depends on your available RAM)
chunk_size = 100_000

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
# TRANSFORM THE TEST SET IN CHUNKS (NO PARTIAL_FIT!)
# ---------------------------------------------
test_nrows = testml.shape[0]
transformed_test_chunks = []

for start in range(0, test_nrows, chunk_size):
    end = min(start + chunk_size, test_nrows)
    
    # Slice out the chunk
    chunk = testml.iloc[start:end].copy()
    
    # Scale using the scaler fitted on TRAIN
    chunk[all_numeric_cols] = scaler.transform(chunk[all_numeric_cols])
    
    # Collect transformed chunk
    transformed_test_chunks.append(chunk)

# Concatenate all transformed chunks
test_scaled = pd.concat(transformed_test_chunks, ignore_index=True)

print("Test scaling done. Final shape:", test_scaled.shape)

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

# Making predictions
y_pred = np.expm1(xgb_model.predict(test_scaled))*testml["square_feet"]


# In[ ]:


import os
from datetime import datetime

submission_df = pd.DataFrame({
    'row_id': row_id,
    'meter_reading': y_pred
})

# Step 5: Ensure the directory exists
output_dir = r'C:\Users\almon\Documents\Learning\Machine Learning\Kaggle\ASHARE\Results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 6: Save the submission DataFrame to CSV with the date included in the filename
current_date = datetime.now().strftime('%m%d')  # Get the current date in 'MMDD' format
filename = f'submission_11_2025_baseline_normalized{current_date}.csv'
submission_df.to_csv(os.path.join(output_dir, filename), index=False)







