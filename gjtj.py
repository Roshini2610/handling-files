import pandas as pd
import matplotlib.pyplot as plt

# Step a) Read test and label files
# Assuming you have downloaded and extracted the files.zip

# Define file paths
test_files = {
    "test": "test.csv",
    "smap_test": "smap_test.csv",
    "msl_test": "msl_test.csv",
    "psm_test": "psm_test.csv"
}

label_files = {
    "test": "test_labels.csv",
    "smap_test": "smap_test_labels.csv",
    "msl_test": "msl_test_labels.csv",
    "psm_test": "psm_test_labels.csv"
}

# Read files into dataframes
test_data = {}
label_data = {}

for key, file in test_files.items():
    test_data[key] = pd.read_csv(file)

for key, file in label_files.items():
    label_data[key] = pd.read_csv(file)

# Step b) Draw time series plots with anomaly regions
def plot_with_anomalies(data, labels, title):
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Time Series Data")
    for i, row in labels.iterrows():
        plt.axvspan(row['start'], row['end'], color='red', alpha=0.3, label="Anomaly Region")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

for key in test_files.keys():
    plot_with_anomalies(test_data[key]['timestamp'], label_data[key], f"Time Series Plot for {key}")

# Step c) Perform EDA and find out root cause
# Your EDA code here

# Step d) Find out the variables which are the root cause for the anomaly
# Your analysis code here

# Step c) Perform EDA and find out root cause

# 1. Summary Statistics
print(test_data['test'].describe())

# 2. Visualize Data Distribution
plt.figure(figsize=(12, 6))
plt.hist(test_data['test']['value'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# 3. Time Series Analysis
# Example: Plot time series and autocorrelation function
plt.figure(figsize=(12, 6))
plt.plot(test_data['test']['timestamp'], test_data['test']['value'], label='Time Series Data')
plt.title('Time Series Plot')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Example of computing autocorrelation
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(test_data['test']['value'], lags=30)
plt.show()

# 4. Anomaly Detection
# Example: Use statistical method (e.g., z-score) for anomaly detection
threshold = 2 # Set a threshold for anomaly detection
test_data['test']['z_score'] = (test_data['test']['value'] - test_data['test']['value'].mean()) / test_data['test']['value'].std()
anomalies = test_data['test'][abs(test_data['test']['z_score']) > threshold]

# Step d) Find out the variables which are the root cause for the anomaly

# 1. Correlation Analysis
correlation_matrix = test_data['test'].corr()
print(correlation_matrix)

# 2. Feature Importance (if using machine learning models)

# 3. Domain Knowledge

# 4. Causal Analysis (if applicable)
