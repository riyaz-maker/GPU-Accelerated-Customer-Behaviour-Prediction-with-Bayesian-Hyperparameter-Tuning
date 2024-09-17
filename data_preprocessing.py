import zipfile
import os

with zipfile.ZipFile('/content/drive/MyDrive/2019-Nov.csv.zip', 'r') as zip_ref:
    zip_ref.extractall()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('2019-Nov.csv')

# Parse event_time as datetime
df['event_time'] = pd.to_datetime(df['event_time'])

# Sort the dataset by user_id and event_time to prepare for time series analysis
df = df.sort_values(by=['user_id', 'event_time'])

# Filter out specific events 
df = df[df['event_type'].isin(['view', 'cart', 'purchase'])]

# Create cumulative interaction counts for each user
df['event_count'] = df.groupby('user_id').cumcount() + 1

# Create a time-difference feature between interactions for each user
df['time_diff'] = df.groupby('user_id')['event_time'].diff().dt.total_seconds().fillna(0)

# Aggregate data to get user-level time series for customer behavior prediction
df_agg = df.groupby(['user_id', 'event_time']).agg({
    'event_count': 'max',
    'time_diff': 'sum'
}).reset_index()

scaler = MinMaxScaler()
df_agg['scaled_time_diff'] = scaler.fit_transform(df_agg[['time_diff']])

user_data = []
user_groups = df_agg.groupby('user_id')
for user_id, group in user_groups:
    user_data.append(group[['scaled_time_diff', 'event_count']].values)

def create_time_series_data(data, window_size=5):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, 0])
        y.append(data[i + window_size, 1])
    return np.array(X), np.array(y)

# Prepare data for all users
X, y = [], []
for data in user_data:
    X_user, y_user = create_time_series_data(data)
    if len(X_user) > 0:
        X.append(X_user)
        y.append(y_user)

# Convert to numpy arrays
X = np.vstack(X)
y = np.concatenate(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]