# Cloud_ML_Compute-resources
optimization of cloud resource usage using Machine Learning
## M21AI564- First Review of MTP2

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

# Step 4: Data Exploration
import pandas as pd
import matplotlib.pyplot as plt


### Data Exploration - Memory Utilization
plt.figure(figsize=(10, 6))
plt.plot(data['Date_Mem Utilization'], data['Memory Utilization'], marker='o', linestyle='-')
plt.xlabel('Date and Time')
plt.ylabel('Memory Utilization')
plt.title('Memory Utilization over Time')
plt.grid(True)
plt.show()

### Data Exploration - CPU Utilization
plt.figure(figsize=(10, 6))
plt.plot(data['Date_CPU Utilization'], data['CPU Utilization'], marker='o', linestyle='-')
plt.xlabel('Date and Time')
plt.ylabel('CPU Utilization')
plt.title('CPU Utilization over Time')
plt.grid(True)
plt.show()

### Data Exploration - Disk Utilization
plt.figure(figsize=(10, 6))
plt.plot(data['Date_Disk Utilization'], data['Disk Utilization'], marker='o', linestyle='-')
plt.xlabel('Date and Time')
plt.ylabel('Disk Utilization')
plt.title('Disk Utilization over Time')
plt.grid(True)
plt.show()



# Step 5: Data Preprocessing and Feature Engineering
#### Data Preprocessing
#### Convert date columns to datetime format with utc=True
data['Date_Mem Utilization'] = pd.to_datetime(data['Date_Mem Utilization'], utc=True)
data['Date_CPU Utilization'] = pd.to_datetime(data['Date_CPU Utilization'], utc=True)
data['Date_Disk Utilization'] = pd.to_datetime(data['Date_Disk Utilization'], utc=True)

### Feature Engineering - Date Components
data['Year_Mem'] = data['Date_Mem Utilization'].dt.year
data['Month_Mem'] = data['Date_Mem Utilization'].dt.month
data['Day_Mem'] = data['Date_Mem Utilization'].dt.day
data['Hour_Mem'] = data['Date_Mem Utilization'].dt.hour
data['Minute_Mem'] = data['Date_Mem Utilization'].dt.minute

data['Year_CPU'] = data['Date_CPU Utilization'].dt.year
data['Month_CPU'] = data['Date_CPU Utilization'].dt.month
data['Day_CPU'] = data['Date_CPU Utilization'].dt.day
data['Hour_CPU'] = data['Date_CPU Utilization'].dt.hour
data['Minute_CPU'] = data['Date_CPU Utilization'].dt.minute

data['Year_Disk'] = data['Date_Disk Utilization'].dt.year
data['Month_Disk'] = data['Date_Disk Utilization'].dt.month
data['Day_Disk'] = data['Date_Disk Utilization'].dt.day
data['Hour_Disk'] = data['Date_Disk Utilization'].dt.hour
data['Minute_Disk'] = data['Date_Disk Utilization'].dt.minute

# Print the first few rows to check the new features
print(data.head())


# Step 6: Model Selection
# a)Linear Regression (Time Series Forecasting)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

#### Select the features and target variable
X = data[['Year_Mem', 'Month_Mem', 'Day_Mem', 'Hour_Mem', 'Minute_Mem']].values
y = data['Memory Utilization'].values

#### Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#### Machine Learning Model - Linear Regression (Time Series Forecasting)
model = LinearRegression()
model.fit(X_train, y_train)

#### Make predictions on the test set
y_pred = model.predict(X_test)

#### Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

### Plot the actual vs. predicted memory utilization
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', marker='o')
plt.plot(y_pred, label='Predicted', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Memory Utilization')
plt.title('Actual vs. Predicted Memory Utilization')
plt.legend()
plt.grid(True)
plt.show()

### output:
Mean Absolute Error (MAE): 31.040568211465136
R-squared (R2): 0.033535602828514643

MAE is 31.040568211465136, which means, on average, the model's predictions deviate from the actual values by approximately 31.04 units.
R2 value is 0.033535602828514643, which suggests that the model explains only about 3.35% of the variance in the data. This indicates that the model's predictions are not very accurate in capturing the underlying patterns in the data.

# b)Time Series Forecasting - SARIMA


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

### Fit SARIMA model
model = sm.tsa.SARIMAX(data['CPU Utilization'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
results = model.fit()

#### Make predictions
forecast = results.get_forecast(steps=len(data))  # Forecast for the entire dataset
forecast_mean = forecast.predicted_mean

#### Plot actual vs. predicted CPU utilization
plt.figure(figsize=(10, 6))
plt.plot(data['CPU Utilization'], label='Actual', marker='o')
plt.plot(forecast_mean, label='Predicted', marker='x')
plt.xlabel('Date')
plt.ylabel('CPU Utilization')
plt.title('Actual vs. Predicted CPU Utilization')
plt.legend()
plt.grid(True)
plt.show()

#### Evaluate the model
mae = mean_absolute_error(data['CPU Utilization'], forecast_mean)
r2 = r2_score(data['CPU Utilization'], forecast_mean)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")


### Mean Absolute Error (MAE): 7.138885717052515
#### R-squared (R2): -0.20020900153555088
MAE is 7.138885717052515, which means, on average, the model's predictions deviate from the actual values by approximately 7.14 units.
R2 value is -0.20020900153555088, which suggests that the model does not fit the data well and performs worse than just using the mean of the target variable for predictions. A negative R2 indicates that the model's predictions are not meaningful and the model is not explaining the variance in the data.¶


# c) Time Series Forecasting - Triple Exponential Smoothing
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, r2_score

#### Drop rows with missing values
data.dropna(inplace=True)

#### Function to perform Triple Exponential Smoothing
def triple_exponential_smoothing(data, column):
    model = sm.tsa.ExponentialSmoothing(data[column], trend='add', seasonal='add', seasonal_periods=24)
    fitted_model = model.fit()
    forecast = fitted_model.fittedvalues
    return forecast

#### Time Series Forecasting - Memory Utilization
data['Forecast_Memory'] = triple_exponential_smoothing(data, 'Memory Utilization')

#### Time Series Forecasting - CPU Utilization
data['Forecast_CPU'] = triple_exponential_smoothing(data, 'CPU Utilization')

#### Time Series Forecasting - Disk Utilization
data['Forecast_Disk'] = triple_exponential_smoothing(data, 'Disk Utilization')

#### Calculate MAE and R2 for Memory Utilization forecast
mae_memory = mean_absolute_error(data['Memory Utilization'], data['Forecast_Memory'])
r2_memory = r2_score(data['Memory Utilization'], data['Forecast_Memory'])
print("Memory Utilization - Mean Absolute Error (MAE):", mae_memory)
print("Memory Utilization - R-squared (R2):", r2_memory)

#### Calculate MAE and R2 for CPU Utilization forecast
mae_cpu = mean_absolute_error(data['CPU Utilization'], data['Forecast_CPU'])
r2_cpu = r2_score(data['CPU Utilization'], data['Forecast_CPU'])
print("CPU Utilization - Mean Absolute Error (MAE):", mae_cpu)
print("CPU Utilization - R-squared (R2):", r2_cpu)

#### Calculate MAE and R2 for Disk Utilization forecast
mae_disk = mean_absolute_error(data['Disk Utilization'], data['Forecast_Disk'])
r2_disk = r2_score(data['Disk Utilization'], data['Forecast_Disk'])
print("Disk Utilization - Mean Absolute Error (MAE):", mae_disk)
print("Disk Utilization - R-squared (R2):", r2_disk)

#### Time Series Visualization - Memory Utilization
plt.figure(figsize=(10, 6))
plt.plot(data['Date_Mem Utilization'], data['Memory Utilization'], label='Actual Memory Utilization', color='blue')
plt.plot(data['Date_Mem Utilization'], data['Forecast_Memory'], label='Forecasted Memory Utilization', color='orange')
plt.xlabel('Date and Time')
plt.ylabel('Memory Utilization')
plt.title('Time Series Forecasting: Memory Utilization')
plt.legend()
plt.grid(True)
plt.show()

#### Time Series Visualization - CPU Utilization
plt.figure(figsize=(10, 6))
plt.plot(data['Date_CPU Utilization'], data['CPU Utilization'], label='Actual CPU Utilization', color='green')
plt.plot(data['Date_CPU Utilization'], data['Forecast_CPU'], label='Forecasted CPU Utilization', color='purple')
plt.xlabel('Date and Time')
plt.ylabel('CPU Utilization')
plt.title('Time Series Forecasting: CPU Utilization')
plt.legend()
plt.grid(True)
plt.show()

#### Time Series Visualization - Disk Utilization
plt.figure(figsize=(10, 6))
plt.plot(data['Date_Disk Utilization'], data['Disk Utilization'], label='Actual Disk Utilization', color='red')
plt.plot(data['Date_Disk Utilization'], data['Forecast_Disk'], label='Forecasted Disk Utilization', color='brown')
plt.xlabel('Date and Time')
plt.ylabel('Disk Utilization')
plt.title('Time Series Forecasting: Disk Utilization')
plt.legend()
plt.grid(True)
plt.show()
