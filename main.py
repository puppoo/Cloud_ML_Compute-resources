import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st

# Function to perform Triple Exponential Smoothing
def triple_exponential_smoothing(data, column):
    model = sm.tsa.ExponentialSmoothing(data[column], trend='add', seasonal='add', seasonal_periods=24)
    fitted_model = model.fit()
    forecast = fitted_model.fittedvalues
    return forecast

# Function to save predicted data to a CSV file
def save_predicted_data_to_csv(data, filename):
    save_path = f'predicted_data/{filename}.csv'
    data.to_csv(save_path, index_label='Date')
    return save_path

# Function to predict next 3 months CPU, Memory, and Disk data
def predict_next_3_months(data, column):
    model = sm.tsa.ExponentialSmoothing(data[column], trend='add', seasonal='add', seasonal_periods=24)
    fitted_model = model.fit()
    next_3_months = fitted_model.forecast(steps=90)  # 90 days for 3 months
    return next_3_months


# Function to plot time series forecasts with DateTime values on x-axis
def plot_time_series_forecast_with_datetime(data, column, forecast_column, title):
    fig = go.Figure()

    # Plot actual data
    fig.add_trace(go.Scatter(x=data['Date_Mem Utilization'], y=data[column], mode='lines', name=f'Actual {column}', line=dict(color='blue')))

    # Plot forecasted data
    fig.add_trace(go.Scatter(x=data['Date_Mem Utilization'], y=data[forecast_column], mode='lines', name=f'Forecasted {column}', line=dict(color='orange')))

    # Customize the x-axis to display DateTime values
    fig.update_layout(title=title, xaxis_title='Date and Time', yaxis_title=column, xaxis=dict(type='category', tickformat='%Y-%m-%d %H:%M'))

    st.plotly_chart(fig)

# Main Streamlit app
def main():
    st.title('Cloud Resource Usage Optimization using Machine Learning')

    # User option for uploading a CSV file
    st.sidebar.title('Upload Custom CSV')
    uploaded_file = st.sidebar.file_uploader('Upload a CSV file', type=['csv'])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)

        # Drop rows with missing values
        data.dropna(inplace=True)

        # Convert the DateTime columns to datetime format with specific format
        date_format = "%Y-%m-%dT%H:%M:%S"
        data['Date_Mem Utilization'] = pd.to_datetime(data['Date_Mem Utilization'], format=date_format, errors='coerce')
        data['Date_CPU Utilization'] = pd.to_datetime(data['Date_CPU Utilization'], format=date_format, errors='coerce')
        data['Date_Disk Utilization'] = pd.to_datetime(data['Date_Disk Utilization'], format=date_format, errors='coerce')

        # Filter out any rows with invalid date values
        data = data.dropna(subset=['Date_Mem Utilization', 'Date_CPU Utilization', 'Date_Disk Utilization'])

        # User option for data granularity
        st.sidebar.title('Data Granularity Options')
        granularity_option = st.sidebar.radio('Select data granularity:', ['Days', 'Months'])

        # Group the data based on user's choice of granularity
        if granularity_option == 'Days':
            data['Date_Mem Utilization'] = pd.to_datetime(data['Date_Mem Utilization'], format=date_format, errors='coerce')
            data['Date_CPU Utilization'] = pd.to_datetime(data['Date_CPU Utilization'], format=date_format, errors='coerce')
            data['Date_Disk Utilization'] = pd.to_datetime(data['Date_Disk Utilization'], format=date_format,
                                                           errors='coerce')
        elif granularity_option == 'Months':
            data['Date_Mem Utilization'] = pd.to_datetime(data['Date_Mem Utilization'], format=date_format,
                                                          errors='coerce').dt.to_period('M').dt.to_timestamp()
            data['Date_CPU Utilization'] = pd.to_datetime(data['Date_CPU Utilization'], format=date_format,
                                                          errors='coerce').dt.to_period('M').dt.to_timestamp()
            data['Date_Disk Utilization'] = pd.to_datetime(data['Date_Disk Utilization'], format=date_format,
                                                           errors='coerce').dt.to_period('M').dt.to_timestamp()

        # User option for grouping the data
        st.sidebar.title('Data Grouping Options')
        option = st.sidebar.selectbox('Select an option for data grouping:',
                                      ['Group by Hostname', 'Group by Region', 'Group by Instance type'])

        # Group the data based on user's choice
        # Group the data based on user's choice
        if option == 'Group by Hostname':
            grouped_data = data.groupby('Hostname',
                                        as_index=False)  # Use 'as_index=False' to keep 'Hostname' as a column
            title_suffix = 'Grouped by Hostname'
            group_column = 'Hostname'
        elif option == 'Group by Region':
            grouped_data = data.groupby('Region', as_index=False)  # Use 'as_index=False' to keep 'Region' as a column
            title_suffix = 'Grouped by Region'
            group_column = 'Region'
        else:
            grouped_data = data.groupby('Instance type',
                                        as_index=False)  # Use 'as_index=False' to keep 'Instance type' as a column
            title_suffix = 'Grouped by Instance type'
            group_column = 'Instance type'

        # Time Series Forecasting and Visualization for each group
        for group_name, group_data in grouped_data:
            # Time Series Forecasting - Memory Utilization
            group_data['Forecast_Memory'] = triple_exponential_smoothing(group_data, 'Memory Utilization')

            # Time Series Forecasting - CPU Utilization
            group_data['Forecast_CPU'] = triple_exponential_smoothing(group_data, 'CPU Utilization')

            # Time Series Forecasting - Disk Utilization
            group_data['Forecast_Disk'] = triple_exponential_smoothing(group_data, 'Disk Utilization')


            # Calculate MAE and R2 for each forecast
            mae_memory = mean_absolute_error(group_data['Memory Utilization'], group_data['Forecast_Memory'])
            r2_memory = r2_score(group_data['Memory Utilization'], group_data['Forecast_Memory'])

            mae_cpu = mean_absolute_error(group_data['CPU Utilization'], group_data['Forecast_CPU'])
            r2_cpu = r2_score(group_data['CPU Utilization'], group_data['Forecast_CPU'])

            mae_disk = mean_absolute_error(group_data['Disk Utilization'], group_data['Forecast_Disk'])
            r2_disk = r2_score(group_data['Disk Utilization'], group_data['Forecast_Disk'])

            # Print MAE and R2 for each forecast
            st.write(f"Group: {group_name}")
            st.write("Memory Utilization - Mean Absolute Error (MAE):", mae_memory)
            st.write("Memory Utilization - R-squared (R2):", r2_memory)

            st.write("CPU Utilization - Mean Absolute Error (MAE):", mae_cpu)
            st.write("CPU Utilization - R-squared (R2):", r2_cpu)

            st.write("Disk Utilization - Mean Absolute Error (MAE):", mae_disk)
            st.write("Disk Utilization - R-squared (R2):", r2_disk)
            st.write("\n")

            # Time Series Visualization - Memory Utilization
            st.subheader(f'Time Series Forecasting: Memory Utilization ({group_name} {title_suffix})')
            plot_time_series_forecast_with_datetime(group_data, 'Memory Utilization', 'Forecast_Memory',
                                                    f'Memory Utilization ({group_name} {title_suffix})')

            # Time Series Visualization - CPU Utilization
            st.subheader(f'Time Series Forecasting: CPU Utilization ({group_name} {title_suffix})')
            plot_time_series_forecast_with_datetime(group_data, 'CPU Utilization', 'Forecast_CPU',
                                                    f'CPU Utilization ({group_name} {title_suffix})')

            # Time Series Visualization - Disk Utilization
            st.subheader(f'Time Series Forecasting: Disk Utilization ({group_name} {title_suffix})')
            plot_time_series_forecast_with_datetime(group_data, 'Disk Utilization', 'Forecast_Disk',
                                                    f'Disk Utilization ({group_name} {title_suffix})')

            st.subheader(f'Predicted Data for Next 3 Months: {group_name} {title_suffix}')
            st.write(f"Group: {group_name}")


            # Predict and print data for the next 3 months for each group
            st.subheader(f'Predicted Data for Next 3 Months: {group_name} {title_suffix}')
            st.write(f"Group: {group_name}")

            # Predict the next 3 months data for each group
            next_3_months_memory = predict_next_3_months(group_data, 'Memory Utilization')
            next_3_months_cpu = predict_next_3_months(group_data, 'CPU Utilization')
            next_3_months_disk = predict_next_3_months(group_data, 'Disk Utilization')

            # Create a DataFrame for the next 3 months' predictions
            next_3_months_predictions = pd.DataFrame({
                'Date_Mem Utilization': pd.date_range(start=group_data['Date_Mem Utilization'].max(), periods=90,
                                                      freq='D'),
                'Memory Utilization': next_3_months_memory,
                'Date_CPU Utilization': pd.date_range(start=group_data['Date_CPU Utilization'].max(), periods=90,
                                                      freq='D'),
                'CPU Utilization': next_3_months_cpu,
                'Date_Disk Utilization': pd.date_range(start=group_data['Date_Disk Utilization'].max(), periods=90,
                                                       freq='D'),
                'Disk Utilization': next_3_months_disk
            })

            # Create a DataFrame for the next 3 months' predictions
            next_3_months_predictions = pd.DataFrame({
                'Date_Mem Utilization': pd.date_range(start=group_data['Date_Mem Utilization'].max(), periods=90,
                                                      freq='D'),
                'Memory Utilization': next_3_months_memory,
                'Date_CPU Utilization': pd.date_range(start=group_data['Date_CPU Utilization'].max(), periods=90,
                                                      freq='D'),
                'CPU Utilization': next_3_months_cpu,
                'Date_Disk Utilization': pd.date_range(start=group_data['Date_Disk Utilization'].max(), periods=90,
                                                       freq='D'),
                'Disk Utilization': next_3_months_disk
            })

            # Display the predicted data for the next 3 months
            st.write(next_3_months_predictions)

            # Time Series Visualization - Memory Utilization
            st.subheader(f'Time Series Forecasting: Memory Utilization ({group_name} {title_suffix})')
            plot_time_series_forecast_with_datetime(group_data, 'Memory Utilization', 'Forecast_Memory',
                                                    f'Memory Utilization ({group_name} {title_suffix})')

            # Time Series Visualization - CPU Utilization
            st.subheader(f'Time Series Forecasting: CPU Utilization ({group_name} {title_suffix})')
            plot_time_series_forecast_with_datetime(group_data, 'CPU Utilization', 'Forecast_CPU',
                                                    f'CPU Utilization ({group_name} {title_suffix})')

            # Time Series Visualization - Disk Utilization
            st.subheader(f'Time Series Forecasting: Disk Utilization ({group_name} {title_suffix})')
            plot_time_series_forecast_with_datetime(group_data, 'Disk Utilization', 'Forecast_Disk',
                                                    f'Disk Utilization ({group_name} {title_suffix})')

            st.download_button(
                label="Download Predicted Data (Next 3 Months)",
                data=next_3_months_predictions.to_csv(index=False).encode('utf-8'),
                file_name=f"Predicted_Data_{group_name}_{title_suffix}.csv",
                mime="text/csv"
            )


# Run the main app
if __name__ == '__main__':
	main()

