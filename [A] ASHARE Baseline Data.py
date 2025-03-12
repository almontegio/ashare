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
RANDOM_STATE = 55


# In[2]:


train_data = pd.read_csv(r"C:\Users\almon\OneDrive\Documents\Learning\Machine Learning\Kaggle\ASHARE\Cleaned\cleaned_train_0904.csv")
test_data = pd.read_csv(r"C:\Users\almon\OneDrive\Documents\Learning\Machine Learning\Kaggle\ASHARE\Cleaned\cleaned_test_0904.csv")


# In[3]:


print(test_data.isna().sum().sort_values(ascending=False).head(20))
print(train_data.head())


# In[4]:


train_data.loc[(train_data['site_id'] == 0) & (train_data['meter'] == 0), 'meter_reading'] *= 0.2931
train_data['meter_reading']= np.log1p(train_data['meter_reading'])

import pandas as pd
import matplotlib.pyplot as plt
import os

# Directory to save figures
save_dir = r"C:\Users\almon\OneDrive\Documents\Learning\Machine Learning\Kaggle\ASHARE\Figures"

# Ensure the directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Assuming train_data is the dataframe
df_filtered = train_data[(train_data['meter'] == 0) & (train_data['site_id'].between(0, 15))]

# Convert timestamp to datetime
df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])

# Loop through site IDs 0 to 15
for site_id in range(0, 16):
    site_data = df_filtered[df_filtered['site_id'] == site_id]
    
    # Group by timestamp and calculate mean of meter readings for aggregation
    aggregated_data = site_data.groupby('timestamp').agg({'meter_reading': 'mean'}).reset_index()
    
    # Create a new figure for each site_id
    plt.figure(figsize=(12, 6))  # Increase the figure size
    
    # Scatter plot with opaque points and size 5
    plt.scatter(aggregated_data['timestamp'], aggregated_data['meter_reading'], color='blue', s=5, alpha=1.0)  # Opaque points with size 5
    
    # Line plot with lower opacity connecting the points
    plt.plot(aggregated_data['timestamp'], aggregated_data['meter_reading'], color='blue', alpha=0.3, linewidth=0.7)  # Low-opacity lines
    
    # Add gridlines for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add title and labels
    plt.title(f'Site ID {site_id} (Average Meter 0 Readings)', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Average Meter Reading', fontsize=12)
    
    # Rotate x-ticks for better visibility
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(save_dir, f'Site_ID_{site_id}_Average_Meter_Readings.png')
    plt.savefig(save_path)  # Save the plot to the specified directory
    
    # Show the plot (optional if you only want to save)
    # plt.show()  # You can comment this out if you don't want to display each plot
import pandas as pd
import matplotlib.pyplot as plt
import os

# Directory to save figures
save_dir = r"C:\Users\almon\OneDrive\Documents\Learning\Machine Learning\Kaggle\ASHARE\Figures"

# Ensure the directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Assuming train_data is the dataframe
df_filtered = train_data[(train_data['site_id'].between(0, 15))]

# Convert timestamp to datetime
df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])

# Loop through site IDs 0 to 15
for site_id in range(0, 16):
    site_data = df_filtered[df_filtered['site_id'] == site_id]
    
    # Group by timestamp and calculate mean of air temperature for aggregation
    aggregated_data = site_data.groupby('timestamp').agg({'air_temperature': 'mean'}).reset_index()
    
    # Create a new figure for each site_id
    plt.figure(figsize=(12, 6))  # Increase the figure size
    
    # Scatter plot with opaque points and size 5
    plt.scatter(aggregated_data['timestamp'], aggregated_data['air_temperature'], color='orange', s=5, alpha=1.0)  # Opaque points with size 5
    
    # Line plot with lower opacity connecting the points
    plt.plot(aggregated_data['timestamp'], aggregated_data['air_temperature'], color='orange', alpha=0.3, linewidth=0.7)  # Low-opacity lines
    
    # Add gridlines for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add title and labels
    plt.title(f'Site ID {site_id} (Average Air Temperature)', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Average Air Temperature (°C)', fontsize=12)
    
    # Rotate x-ticks for better visibility
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(save_dir, f'Site_ID_{site_id}_Average_Air_Temperature.png')
    plt.savefig(save_path)  # Save the plot to the specified directory
    
    # Show the plot (optional if you only want to save)
    # plt.show()  # You can comment this out if you don't want to display each plot
import pandas as pd
import matplotlib.pyplot as plt
import os

# Directory to save figures
save_dir = r"C:\Users\almon\OneDrive\Documents\Learning\Machine Learning\Kaggle\ASHARE\Figures"

# Ensure the directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Assuming train_data is the dataframe
df_filtered = train_data[(train_data['site_id'].between(0, 15))]

# Convert timestamp to datetime
df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])

# Features to plot
features = ['cloud_coverage', 'dew_temperature', 'wind_direction', 'wind_speed']

# Loop through site IDs 0 to 15
for site_id in range(0, 16):
    site_data = df_filtered[df_filtered['site_id'] == site_id]
    
    # Group by timestamp and calculate mean of each feature for aggregation
    aggregated_data = site_data.groupby('timestamp').agg({
        'cloud_coverage': 'mean',
        'dew_temperature': 'mean',
        'wind_direction': 'mean',
        'wind_speed': 'mean'
    }).reset_index()
    
    # Create a 2x2 subplot for each site_id
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 grid of subplots
    
    # Plot each feature in a subplot
    for i, feature in enumerate(features):
        row, col = divmod(i, 2)  # Determine the position in the 2x2 grid
        ax = axes[row, col]  # Access the correct subplot
        
        # Scatter plot with opaque points and size 5
        ax.scatter(aggregated_data['timestamp'], aggregated_data[feature], color='blue', s=5, alpha=1.0)
        
        # Line plot with lower opacity connecting the points
        ax.plot(aggregated_data['timestamp'], aggregated_data[feature], color='blue', alpha=0.3, linewidth=0.7)
        
        # Add title and labels
        ax.set_title(f'{feature.replace("_", " ").capitalize()} (Site ID {site_id})', fontsize=14)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(f'Average {feature.replace("_", " ").capitalize()}', fontsize=12)
        
        # Rotate x-ticks for better visibility
        ax.tick_params(axis='x', rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(save_dir, f'Site_ID_{site_id}_Weather_Features.png')
    plt.savefig(save_path)  # Save the plot to the specified directory
    
    # Show the plot (optional if you only want to save)
    # plt.show()  # You can comment this out if you don't want to display each plot

# In[5]:


features = ['floor_count', 'air_temperature', 'cloud_coverage', 'dew_temperature', 
            'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']

# Filter for site_id between 0 and 15
df_filtered = train_data[(train_data['site_id'].between(0, 15))]

# Group by site_id and find the minimum values for the specified features
min_values_per_site = df_filtered.groupby('site_id')[features].min().reset_index()

# Display the result
print(min_values_per_site)


# # Data Cleaning (Weather Data)

# In[6]:


import pandas as pd
import numpy as np

# Columns to check and replace minimum values
columns_to_check = ['floor_count', 'year_built', 'cloud_coverage', 'precip_depth_1_hr', 
                    'wind_direction', 'sea_level_pressure', 'wind_speed', 'dew_temperature', 'air_temperature']

# Replace the minimum values in each column with NaN
for column in columns_to_check:
    min_value_train = train_data[column].min()
    min_value_test = test_data[column].min()
    train_data[column] = train_data[column].replace(min_value_train, np.nan)
    test_data[column] = test_data[column].replace(min_value_test, np.nan)


# In[7]:


print(train_data.isna().sum().sort_values(ascending=False).head(20))
print(test_data.isna().sum().sort_values(ascending=False).head(20))


# In[8]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # This is required to use IterativeImputer
from sklearn.impute import SimpleImputer, IterativeImputer
from datetime import datetime

# Transformer for imputing missing values with the median
class MedianImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        imputer = SimpleImputer(strategy='median')  # Using median strategy
        X[self.columns] = pd.DataFrame(imputer.fit_transform(X[self.columns]), columns=self.columns)
        return X

# Transformer for linear imputation using Iterative Imputer
class LinearImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        imputer = IterativeImputer(max_iter=10, random_state=0)
        X[self.columns] = pd.DataFrame(imputer.fit_transform(X[self.columns]), columns=self.columns)
        return X

class YearBinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, year_columns):
        self.year_columns = year_columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        bins = [-1,0,1920,1950, 1980, 2005, float('inf')]
        labels = ['Null','<1920','<1950','<1980', '1981-2005', '2006 onwards']
        for col in self.year_columns:
            X[col + '_binned'] = pd.cut(X[col], bins=bins, labels=labels)
            X.drop(columns=[col], inplace=True)  # Drop original year column
        return X
    
class TimestampTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            # Convert timestamp strings to datetime objects and then to Unix timestamps
            X[col] = X[col].apply(lambda x: int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp()))
        return X

# Function to create the custom pipeline
def create_custom_pipeline():
    pipeline = Pipeline([
        # Step 1: Impute missing values with the median
        ('median_imputer', MedianImputerTransformer(columns=['floor_count', 'cloud_coverage', 'precip_depth_1_hr', 
                                                             'wind_direction', 'wind_speed', 'year_built', 'sea_level_pressure'])),
        
        # Step 2: Perform linear imputation (Iterative Imputer) for specific columns
        ('linear_imputer', LinearImputerTransformer(columns=['dew_temperature', 'air_temperature'])),
        ('year_binning', YearBinningTransformer(year_columns=['year_built'])),
        ('date_time', TimestampTransformer(columns=['timestamp']))
    ])
    

    
    return pipeline

# Example usage:
pipeline = create_custom_pipeline()

# Assuming `train_data` is your DataFrame
train_data_clean = pipeline.fit_transform(train_data)
test_data_clean = pipeline.transform(test_data)


# In[9]:


print(train_data_clean.isna().sum().sort_values(ascending=False).head(20))
print(test_data_clean.isna().sum().sort_values(ascending=False).head(20))


# In[10]:


# Use sparse=True to reduce memory usage
train_data_clean = pd.get_dummies(train_data_clean, drop_first=False, dtype='int')
test_data_clean = pd.get_dummies(test_data_clean, drop_first=False, dtype='int')



# In[11]:


train_data_clean.info()
test_data_clean.info()


# In[12]:


columns_in_train_not_in_test = [col for col in train_data_clean if col not in test_data_clean]

# Find columns in ctest_cat but not in ctrain_cat
columns_in_test_not_in_train = [col for col in test_data_clean if col not in train_data_clean]

# Print the results
print("Columns in train but not in test:", columns_in_train_not_in_test)
print("Columns in test but not in train:", columns_in_test_not_in_train)


# In[37]:


row_id = test_data_clean["row_id"]


# In[13]:


print(train_data_clean.shape)


# In[14]:


#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            print("min for this col: ",mn)
            print("max for this col: ",mx)
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.001 and result < 0.001:
                IsInt = True            
            # Make Integer/unsigned Integer datatypes
            if IsInt == True:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist


# In[15]:


clean_data_min, NAlist = reduce_mem_usage(train_data_clean)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)


# In[16]:


clean_testdata_min, NAlist = reduce_mem_usage(test_data_clean)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)


# In[18]:


train = clean_data_min
test = clean_testdata_min


# In[19]:


train["timestamp"] = pd.to_datetime(train["timestamp"])
train["hour"] = train["timestamp"].dt.hour
train["day"] = train["timestamp"].dt.day
train["weekend"] = train["timestamp"].dt.weekday
train["month"] = train["timestamp"].dt.month


# In[20]:


test["timestamp"] = pd.to_datetime(test["timestamp"])
test["hour"] = test["timestamp"].dt.hour
test["day"] = test["timestamp"].dt.day
test["weekend"] = test["timestamp"].dt.weekday
test["month"] = test["timestamp"].dt.month


# In[21]:


train = train.drop("timestamp", axis = 1)
test = test.drop("timestamp", axis = 1)


# In[22]:


drop_cols = ["precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed", "floor_count"]
train = train.drop(drop_cols, axis = 1)
test = test.drop(drop_cols, axis = 1)


# In[23]:


target = np.log1p(train["meter_reading"])


# In[24]:


train = train.drop("meter_reading", axis = 1)
test = test.drop("row_id", axis = 1)
columns_in_train_not_in_test = [col for col in train if col not in test]

# Find columns in ctest_cat but not in ctrain_cat
columns_in_test_not_in_train = [col for col in test if col not in train]

# Print the results
print("Columns in train but not in test:", columns_in_train_not_in_test)
print("Columns in test but not in train:", columns_in_test_not_in_train)


# In[25]:


print(train.info())


# In[26]:


train, NAlist = reduce_mem_usage(train)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)


# In[27]:


test, NAlist = reduce_mem_usage(test)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)


# In[30]:


def clean_column_names(df):
    df.columns = [col.replace('<', '').replace('>', '') for col in df.columns]
    return df

train = clean_column_names(train)
test = clean_column_names(test)


# In[31]:


print(train.info())


# In[41]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Features and target
X = train
y = target

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining the XGBoost regressor model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',  # For regression tasks
    n_estimators=100,              # Number of trees
    learning_rate=0.1,             # Learning rate
    max_depth=6,                   # Maximum tree depth
    random_state=42                # Seed for reproducibility
)

# Training the model
xgb_model.fit(X_train, y_train)

# Making predictions
y_pred = np.exp(xgb_model.predict(test))



# In[39]:


import os
from datetime import datetime

submission_df = pd.DataFrame({
    'row_id': row_id,
    'meter_reading': y_pred
})

# Step 5: Ensure the directory exists
output_dir = r'C:\Users\almon\OneDrive\Documents\Learning\Machine Learning\Kaggle\ASHARE\Results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 6: Save the submission DataFrame to CSV with the date included in the filename
current_date = datetime.now().strftime('%m%d')  # Get the current date in 'MMDD' format
filename = f'submission_2_xgb_baseline{current_date}.csv'
submission_df.to_csv(os.path.join(output_dir, filename), index=False)

import datetime

# Get the current date
date_string = datetime.datetime.now().strftime("%m%d")

# Save the data to the new filenames
datatrain = clean_data_min
datatest = clean_testdata_min

# Create the file names with the date appended
train_filename = fr'C:\Users\almon\OneDrive\Documents\Learning\Machine Learning\Kaggle\ASHARE\Cleaned\train_clean_{date_string}.csv'
test_filename = fr'C:\Users\almon\OneDrive\Documents\Learning\Machine Learning\Kaggle\ASHARE\Cleaned\test_clean_{date_string}.csv'

datatrain.to_csv(train_filename, index=False)
datatest.to_csv(test_filename, index=False)
# In[ ]:





# In[ ]:




