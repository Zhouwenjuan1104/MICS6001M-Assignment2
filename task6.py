import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

# Remove linear trend
# Generate synthetic series with a linear trend
t = np.arange(100)
linear_trend = 0.5 * t  # Linear trend
series = linear_trend + np.random.normal(0, 1, 100)  # Add noise

# ADF test for original series
linear_adf_result = adfuller(series)
print('Original ADF Statistic:', linear_adf_result[0])
print('Original p-value:', linear_adf_result[1])
print()

# Differencing
diff_series = np.diff(series)

# ADF test for differenced series
linear_adf_result_diff = adfuller(diff_series)
print('Differenced ADF Statistic:', linear_adf_result_diff[0])
print('Differenced p-value:', linear_adf_result_diff[1])
print()

# Plot
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(series, label="Original Series")
plt.title("Before Differencing")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(diff_series, label="Differenced Series")
plt.title("After Differencing")
plt.legend()
plt.savefig('pics/remove_linear.png', dpi=350)
plt.close()

# Remove exponential trend

# Generate synthetic series with an exponential trend
exponential_trend = np.exp(0.05 * t)  # Exponential trend
series_exp = exponential_trend + np.random.normal(0, 1, 100)  # Add noise

# Log transformation
log_series = np.log(series_exp)

# Convert to pandas Series
log_series = pd.Series(log_series)

# Differencing
diff_log_series = log_series.diff().dropna()

# ADF test
exponential_adf_result = adfuller(series_exp)
print('Exponential ADF Statistic:', exponential_adf_result[0])
print('Exponential p-value:', exponential_adf_result[1])
print()

log_adf_result = adfuller(log_series.dropna())
print('Log ADF Statistic:', log_adf_result[0])
print('Log p-value:', log_adf_result[1])
print()

diff_log_adf_result = adfuller(diff_log_series)
print('Differenced Log ADF Statistic:', diff_log_adf_result[0])
print('Differenced Log p-value:', diff_log_adf_result[1])
print()

# Plot
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(series_exp, label="Original Series")
plt.title("Before Transformation")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(log_series, label="Log Transformed Series")
plt.title("After Log Transformation")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(diff_log_series, label="Differenced Log Series")
plt.title("After Differencing")
plt.legend()
plt.savefig('pics/remove_exponential.png', dpi=350)
# plt.show()
plt.close()

# Remove seasonal trend

# Generate synthetic series with a seasonal trend
np.random.seed(42)
t = np.arange(100)
seasonal_period = 12  # Seasonal period
seasonal_trend = 15 * np.sin(2 * np.pi * t / seasonal_period)  # Seasonal trend
noise = np.random.normal(0, 1, 100)  # Add noise
series = seasonal_trend + noise

# Differencing with different orders
diff_orders = [1, 2, 5, 6, 8, 12]  # Different differencing orders

for order in diff_orders:
    diff_series = np.diff(series, n=order)
    
    plt.figure(figsize=(14, 10))
    
    # Plot differenced series
    plt.subplot(2, 1, 1)
    plt.plot(diff_series, label=f"Diff Order={order}")
    plt.title(f"Differenced Series (Order={order})")
    plt.legend()
    plt.grid(True)
    
    # Plot ACF of differenced series
    plt.subplot(2, 1, 2)
    plot_acf(diff_series, lags=24, ax=plt.gca(), alpha=0.05, zero=False)
    plt.title(f"ACF (Order={order})")
    plt.grid(True)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'pics/differenced_series_acf_order_{order}.png', dpi=350)
    plt.close()

# ADF test for differenced series

for order in diff_orders:
    diff_series = np.diff(series, n=order)
    adf_result = adfuller(diff_series)
    print(f"ADF Test (Order={order}):")
    # print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    # print(f"Critical Values: {adf_result[4]}")
    print()
####################
