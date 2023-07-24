# Cloud_ML_Compute-resources
optimization of cloud resource usage using Machine Learning

# Dataset Gathering :Collection and Pre-processing
Sample AWS Test Dataset used:![image](https://github.com/puppoo/Cloud_ML_Compute-resources/assets/39239635/8a72d0df-c40e-4556-91f9-70feb6968abd)

# Data Collection and Preprocessing:
import pandas as pd

# Step 1: Data Collection
# Assuming you have a CSV file named 'cloud_resource_usage.csv' with columns like 'Date_Mem Utilization', 'Date_CPU Utilization', 'Date_Disk Utilization', etc.
# Replace 'cloud_resource_usage.csv' with the actual filename and appropriate path
data = pd.read_csv('performance_data.csv')

# Step 2: Data Preprocessing
# Handle missing values
data.dropna(inplace=True)  # Drop rows with any missing values

# Handle outliers (optional, if needed)
# You can use various statistical techniques or domain knowledge to detect and handle outliers.

# Step 3: Data Transformation
# Convert 'Date_Mem Utilization', 'Date_CPU Utilization', and 'Date_Disk Utilization' columns to datetime format
data['Date_Mem Utilization'] = pd.to_datetime(data['Date_Mem Utilization'])
data['Date_CPU Utilization'] = pd.to_datetime(data['Date_CPU Utilization'])
data['Date_Disk Utilization'] = pd.to_datetime(data['Date_Disk Utilization'])

# Optionally, set 'Date_Mem Utilization' or other date columns as the DataFrame index for time series analysis
# data.set_index('Date_Mem Utilization', inplace=True)
# data.set_index('Date_CPU Utilization', inplace=True)
# data.set_index('Date_Disk Utilization', inplace=True)

# Further data transformation or feature engineering can be performed here based on the analysis goals.

# Print the first few rows to check the preprocessed data
print(data.head())
