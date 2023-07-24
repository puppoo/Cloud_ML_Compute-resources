# Cloud_ML_Compute-resources
optimization of cloud resource usage using Machine Learning

# Dataset Gathering :Collection and Pre-processing
Sample AWS Test Dataset used:![image](https://github.com/puppoo/Cloud_ML_Compute-resources/assets/39239635/8a72d0df-c40e-4556-91f9-70feb6968abd)


# Step 1: Data Collection¶
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('performance_data.csv')

# Step 2: Data Preprocessing and Handling Outliers
### Data Preprocessing
### Convert date columns to datetime format
date_columns = ['Date_Mem Utilization', 'Date_CPU Utilization', 'Date_Disk Utilization']
for col in date_columns:
    df[col] = pd.to_datetime(df[col])

### Handling Outliers - Detect and Remove/Adjust Outliers
#### consider using the Z-score method to detect outliers

def detect_outliers_zscore(data, threshold=3):
    z_scores = (data - data.mean()) / data.std()
    return abs(z_scores) > threshold

### Detect outliers for CPU, Memory, and Disk Utilization columns
outliers_cpu = detect_outliers_zscore(df['CPU Utilization'])
outliers_memory = detect_outliers_zscore(df['Memory Utilization'])
outliers_disk = detect_outliers_zscore(df['Disk Utilization'])

### Remove or replace outliers (e.g., replace with mean, median, or drop the rows)
df['CPU Utilization'][outliers_cpu] = df['CPU Utilization'].mean()
df['Memory Utilization'][outliers_memory] = df['Memory Utilization'].mean()
df['Disk Utilization'][outliers_disk] = df['Disk Utilization'].mean()

# Visualize Outliers with respect to Hostname and Regions

### Plotting CPU Utilization Outliers
plt.figure(figsize=(12, 6))
plt.scatter(df[outliers_cpu]['Hostname'], df[outliers_cpu]['CPU Utilization'], c='red', label='Outliers')
plt.scatter(df[~outliers_cpu]['Hostname'], df[~outliers_cpu]['CPU Utilization'], c='blue', label='Non-Outliers')
plt.xlabel('Hostname')
plt.ylabel('CPU Utilization')
plt.title('CPU Utilization Outliers')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

### Plotting CPU Utilization Outliers
plt.figure(figsize=(12, 6))
plt.scatter(df[outliers_cpu]['Region'], df[outliers_cpu]['CPU Utilization'], c='red', label='Outliers')
plt.scatter(df[~outliers_cpu]['Region'], df[~outliers_cpu]['CPU Utilization'], c='blue', label='Non-Outliers')
plt.xlabel('Region')
plt.ylabel('CPU Utilization')
plt.title('CPU Utilization Outliers')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


### Plotting Memory Utilization Outliers
plt.figure(figsize=(12, 6))
plt.scatter(df[outliers_memory]['Hostname'], df[outliers_memory]['Memory Utilization'], c='red', label='Outliers')
plt.scatter(df[~outliers_memory]['Hostname'], df[~outliers_memory]['Memory Utilization'], c='blue', label='Non-Outliers')
plt.xlabel('Hostname')
plt.ylabel('Memory Utilization')
plt.title('Memory Utilization Outliers')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

### Plotting Memory Utilization Outliers
plt.figure(figsize=(12, 6))
plt.scatter(df[outliers_memory]['Region'], df[outliers_memory]['Memory Utilization'], c='red', label='Outliers')
plt.scatter(df[~outliers_memory]['Region'], df[~outliers_memory]['Memory Utilization'], c='blue', label='Non-Outliers')
plt.xlabel('Region')
plt.ylabel('Memory Utilization')
plt.title('Memory Utilization Outliers')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

### Plotting Disk Utilization Outliers
plt.figure(figsize=(12, 6))
plt.scatter(df[outliers_disk]['Region'], df[outliers_disk]['Disk Utilization'], c='red', label='Outliers')
plt.scatter(df[~outliers_disk]['Region'], df[~outliers_disk]['Disk Utilization'], c='blue', label='Non-Outliers')
plt.xlabel('Region')
plt.ylabel('Disk Utilization')
plt.title('Disk Utilization Outliers')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


### Plotting Disk Utilization Outliers
plt.figure(figsize=(12, 6))
plt.scatter(df[outliers_disk]['Hostname'], df[outliers_disk]['Disk Utilization'], c='red', label='Outliers')
plt.scatter(df[~outliers_disk]['Hostname'], df[~outliers_disk]['Disk Utilization'], c='blue', label='Non-Outliers')
plt.xlabel('Hostname')
plt.ylabel('Disk Utilization')
plt.title('Disk Utilization Outliers')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Step3: Data Transformation
### Convert 'Date_Mem Utilization', 'Date_CPU Utilization', and 'Date_Disk Utilization' columns to datetime format
data['Date_Mem Utilization'] = pd.to_datetime(data['Date_Mem Utilization'])
data['Date_CPU Utilization'] = pd.to_datetime(data['Date_CPU Utilization'])
data['Date_Disk Utilization'] = pd.to_datetime(data['Date_Disk Utilization'])


print(data.head())
