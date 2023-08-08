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
def plot_time_series_forecast_with_datetime(data, date_column, column, forecast_column, title):
    fig = go.Figure()

    # Sort the data by the date column in chronological order
    data = data.sort_values(by=date_column)

    # Plot actual data
    fig.add_trace(
        go.Scatter(x=data[date_column], y=data[column], mode='lines', name=f'Actual {column}', line=dict(color='blue')))

    # Plot forecasted data
    fig.add_trace(go.Scatter(x=data[date_column], y=data[forecast_column], mode='lines', name=f'Forecasted {column}',
                             line=dict(color='orange')))

    # Customize the x-axis to display DateTime values
    fig.update_layout(title=title, xaxis_title='Date and Time', yaxis_title=column,
                      xaxis=dict(type='category', tickformat='%Y-%m-%d %H:%M'))

    st.plotly_chart(fig)


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

# Function to display average utilization in pie chart form
def display_average_utilization_pie_chart(data, grouping_option):
    avg_memory_utilization = data['Memory Utilization'].mean()
    avg_cpu_utilization = data['CPU Utilization'].mean()
    avg_disk_utilization = data['Disk Utilization'].mean()

    fig = go.Figure(data=[go.Pie(labels=['Memory Utilization', 'CPU Utilization', 'Disk Utilization'],
                                 values=[avg_memory_utilization, avg_cpu_utilization, avg_disk_utilization])])
    fig.update_layout(title=f'Average Utilization - {grouping_option}', showlegend=False)

    st.plotly_chart(fig)

def calculate_average_utilization(data, metric_column):
    average_utilization = data.groupby('Hostname')[metric_column].mean()
    return average_utilization


# Function to plot bar chart
def plot_bar_chart(x, y, x_label, y_label, title):
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
              'rgb(148, 103, 189)', 'rgb(140, 86, 75)', 'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
              'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

    fig = go.Figure()
    for i, (x_val, y_val) in enumerate(zip(x, y)):
        fig.add_trace(go.Bar(x=[x_val], y=[y_val], marker_color=colors[i % len(colors)], name=x_val))

    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, showlegend=True)
    st.plotly_chart(fig)


# Function to gather user feedback
def gather_user_feedback():
    feedback_text = st.sidebar.text_area("Please provide your feedback:", "")
    if st.sidebar.button("Submit Feedback"):
        # Here you can save the feedback or take any other actions
        st.sidebar.success("Thank you for your feedback!")

# Function to save feedback to a text file
def save_feedback(feedback):
    with open("user_feedback.txt", "a") as file:
        file.write(feedback + "\n")


# Main Streamlit app
def main():
    # Set Streamlit page config and theme
    st.set_page_config(page_title='Cloud Resource Usage Optimization', page_icon=':chart_with_upwards_trend:',
                       layout='wide')
    custom_css = """
             body {
                background-color: #f2f2f2; /* Light gray background color */
            }
            .stApp {
                background-color: #d9d9d9; /* Light gray background color */
            }
            .stButton>button {
                background-color: #3498db;
            }
    """
<<<<<<< HEAD

=======
    
>>>>>>> 88e217f56c1a2cfc2e531e323d61970036b6b55e
    st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)

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
        data['Date_Disk Utilization'] = pd.to_datetime(data['Date_Disk Utilization'], format=date_format,
                                                       errors='coerce')

        # Filter out any rows with invalid date values
        data = data.dropna(subset=['Date_Mem Utilization', 'Date_CPU Utilization', 'Date_Disk Utilization'])

        # User option for data granularity
        st.sidebar.title('Data Granularity Options')
        granularity_option = st.sidebar.radio('Select data granularity:', ['Days', 'Months'])

        # Group the data based on user's choice of granularity
        if granularity_option == 'Days':
            date_column = 'Date_Mem Utilization'
        elif granularity_option == 'Months':
            data['Date_Mem Utilization'] = pd.to_datetime(data['Date_Mem Utilization'], format=date_format,
                                                          errors='coerce').dt.to_period('M').dt.to_timestamp()
            date_column = 'Date_Mem Utilization'

        # User option for grouping the data
        st.sidebar.title('Data Grouping Options')
        option = st.sidebar.selectbox('Select an option for data grouping:',
                                      ['Group by Account ID', 'Group by Account name', 'Group by Hostname',
                                       'Group by Instance type', 'Group by OS', 'Group by Region'])



        # Group the data based on user's choice
        if option == 'Group by Account ID':
            grouped_data = data.groupby('Account ID', as_index=False)
            title_suffix = 'Grouped by Account ID'
        elif option == 'Group by Account name':
            grouped_data = data.groupby('Account name', as_index=False)
            title_suffix = 'Grouped by Account name'
        elif option == 'Group by Hostname':
            grouped_data = data.groupby('Hostname', as_index=False)
            title_suffix = 'Grouped by Hostname'
        elif option == 'Group by Instance type':
            grouped_data = data.groupby('Instance type', as_index=False)
            title_suffix = 'Grouped by Instance type'
        elif option == 'Group by OS':
            grouped_data = data.groupby('OS', as_index=False)
            title_suffix = 'Grouped by OS'
        else:
            grouped_data = data.groupby('Region', as_index=False)
            title_suffix = 'Grouped by Region'

        if option and grouped_data:
            # Display average utilization in pie chart form
            display_average_utilization_pie_chart(data, title_suffix)

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
            plot_time_series_forecast_with_datetime(group_data, date_column, 'Memory Utilization', 'Forecast_Memory',
                                                    f'Memory Utilization ({group_name} {title_suffix})')

            # Time Series Visualization - CPU Utilization
            st.subheader(f'Time Series Forecasting: CPU Utilization ({group_name} {title_suffix})')
            plot_time_series_forecast_with_datetime(group_data, date_column, 'CPU Utilization', 'Forecast_CPU',
                                                    f'CPU Utilization ({group_name} {title_suffix})')

            # Time Series Visualization - Disk Utilization
            st.subheader(f'Time Series Forecasting: Disk Utilization ({group_name} {title_suffix})')
            plot_time_series_forecast_with_datetime(group_data, date_column, 'Disk Utilization', 'Forecast_Disk',
                                                    f'Disk Utilization ({group_name} {title_suffix})')

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

            # Display the predicted data for the next 3 months
            st.write(next_3_months_predictions)

            st.download_button(
                label="Download Predicted Data (Next 3 Months)",
                data=next_3_months_predictions.to_csv(index=False).encode('utf-8'),
                file_name=f"Predicted_Data_{group_name}_{title_suffix}.csv",
                mime="text/csv"
            )

            # Calculate average utilization for each metric
            average_memory_utilization = calculate_average_utilization(data, 'Memory Utilization')
            average_cpu_utilization = calculate_average_utilization(data, 'CPU Utilization')
            average_disk_utilization = calculate_average_utilization(data, 'Disk Utilization')

            # Plot bar charts for each metric against the hostname
            st.subheader('Average Memory Utilization by Hostname')
            plot_bar_chart(average_memory_utilization.index, average_memory_utilization.values, 'Hostname',
                           'Average Memory Utilization', 'Average Memory Utilization by Hostname')

            st.subheader('Average CPU Utilization by Hostname')
            plot_bar_chart(average_cpu_utilization.index, average_cpu_utilization.values, 'Hostname',
                           'Average CPU Utilization', 'Average CPU Utilization by Hostname')

            st.subheader('Average Disk Utilization by Hostname')
            plot_bar_chart(average_disk_utilization.index, average_disk_utilization.values, 'Hostname',
                           'Average Disk Utilization', 'Average Disk Utilization by Hostname')

        gather_user_feedback()
        # Run the main app
if __name__ == '__main__':
    main()
