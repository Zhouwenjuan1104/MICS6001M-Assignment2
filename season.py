import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import os
from statsmodels.tsa.stattools import adfuller

# Create directory for saving plots
if not os.path.exists('pics'):
    os.makedirs('pics')

# Generate synthetic series with a seasonal trend
np.random.seed(42)
t = np.arange(200)
seasonal_period = 12  # Seasonal period
seasonal_trend = 10 * np.sin(2 * np.pi * t / seasonal_period)  # Strong seasonal trend
noise = np.random.normal(0, 1, 200)  # Add noise
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

# for order in diff_orders:
#     diff_series = np.diff(series, n=order)
    
#     # Plot differenced series and ACF
#     plt.figure(figsize=(14, 15))
#     plt.subplot(2, 1, 1)
#     plt.plot(diff_series, label=f"Diff Order={order}")
#     plt.title(f"Differenced Series (Order={order})")
#     plt.legend()
#     plt.grid(True)

#     plt.subplot(2, 1, 2)
#     plot_acf(diff_series, lags=24)
#     plt.title(f"ACF (Order={order})")
#     plt.grid(True)
#     plt.savefig(f'pics/differenced_series_order_{order}.png', dpi=350)
#     plt.close()

    # # Plot differenced series
    # plt.figure(figsize=(10, 6))
    # plt.plot(diff_series, label=f"Diff Order={order}")
    # plt.title(f"Differenced Series (Order={order})")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f'pics/differenced_series_order_{order}.png', dpi=350)
    # plt.close()

    # # Plot ACF of differenced series
    # plt.figure(figsize=(10, 6))
    # plot_acf(diff_series, lags=24)
    # plt.title(f"ACF (Order={order})")
    # plt.grid(True)
    # plt.savefig(f'pics/acf_order_{order}.png', dpi=350)
    # plt.close()