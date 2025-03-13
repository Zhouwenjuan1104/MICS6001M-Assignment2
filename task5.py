import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
import numpy as np
import os

# Create directory for saving plots
if not os.path.exists('pics'):
    os.makedirs('pics')

# Load the data
data = pd.read_csv('data.csv')
data.set_index('Year', inplace=True)

# Randomness test
# Line plot
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Temperature change'], marker='o')
plt.title('Temperature Change Over Years')
plt.xlabel('Year')
plt.ylabel('Temperature Change')
plt.grid(True)
plt.savefig('pics/line_plot.png', dpi=350)
plt.close()

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data['Temperature change'], kde=True)
plt.title('Histogram of Temperature Change')
plt.xlabel('Temperature Change')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('pics/histogram.png', dpi=350)
plt.close()

# Density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data['Temperature change'], shade=True)
plt.title('Density Plot of Temperature Change')
plt.xlabel('Temperature Change')
plt.ylabel('Density')
plt.grid(True)
plt.savefig('pics/density_plot.png', dpi=350)
plt.close()

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data['Temperature change'].values.reshape(-1, 1), annot=False, cmap='coolwarm', cbar=True)
plt.title('Heatmap of Temperature Change')
plt.xlabel('Temperature Change')
plt.ylabel('Year')
plt.savefig('pics/heatmap.png', dpi=350)
plt.close()

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data['Temperature change'])
plt.title('Box Plot of Temperature Change')
plt.xlabel('Temperature Change')
plt.grid(True)
plt.savefig('pics/box_plot.png', dpi=350)
plt.close()

# Lag-1 plot
plt.figure(figsize=(6, 6))
sns.regplot(x=data['Temperature change'][:-1], y=data['Temperature change'][1:], ci=None, fit_reg=False)
plt.title('Lag-1 Plot')
plt.xlabel('Temperature Change (t)')
plt.ylabel('Temperature Change (t+1)')
plt.grid(True)
plt.savefig('pics/lag1_plot.png', dpi=350)
plt.close()

# Lag-2 plot
plt.figure(figsize=(6, 6))
sns.regplot(x=data['Temperature change'][:-2], y=data['Temperature change'][2:], ci=None, fit_reg=False)
plt.title('Lag-2 Plot')
plt.xlabel('Temperature Change (t)')
plt.ylabel('Temperature Change (t+2)')
plt.grid(True)
plt.savefig('pics/lag2_plot.png', dpi=350)
plt.close()

# Ljung-Box test
ljung_box_result = acorr_ljungbox(data['Temperature change'], lags=[1], return_df=True)
print("Original Ljung-Box Test Result:")
print(ljung_box_result)

# Stationarity test
# ADF test
adf_result = adfuller(data['Temperature change'])
print('Original ADF Statistic:', adf_result[0])
print('Original p-value:', adf_result[1])

# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(data['Temperature change'], ax=plt.gca(), lags=40)
plt.title('Autocorrelation Function')
plt.subplot(122)
plot_pacf(data['Temperature change'], ax=plt.gca(), lags=40)
plt.title('Partial Autocorrelation Function')
plt.savefig('pics/acf_pacf.png', dpi=350)
plt.close()

# First order difference
data['Temperature change_diff'] = data['Temperature change'].diff()

# Lung-Box test after first order difference
ljung_box_result_diff = acorr_ljungbox(data['Temperature change_diff'].dropna(), lags=[1], return_df=True)
print("Ljung-Box Test Result after First Order Difference:")
print(ljung_box_result_diff)

# ADF test after first order difference
adf_result_diff = adfuller(data['Temperature change_diff'].dropna())
print('First order ADF Statistic:', adf_result_diff[0])
print('First order p-value:', adf_result_diff[1])

# Plot ACF and PACF after first order difference
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(data['Temperature change_diff'].dropna(), ax=plt.gca(), lags=40)
plt.title('Autocorrelation Function (First Order Difference)')
plt.subplot(122)
plot_pacf(data['Temperature change_diff'].dropna(), ax=plt.gca(), lags=40)
plt.title('Partial Autocorrelation Function (First Order Difference)')
plt.savefig('pics/acf_pacf_diff1.png', dpi=350)
plt.close()

# Second order difference
data['Temperature change_diff2'] = data['Temperature change_diff'].diff()

# Ljung-Box test after second order difference
ljung_box_result_diff2 = acorr_ljungbox(data['Temperature change_diff2'].dropna(), lags=[1], return_df=True)
print("Ljung-Box Test Result after Second Order Difference:")
print(ljung_box_result_diff2)

# ADF test after second order difference
adf_result_diff2 = adfuller(data['Temperature change_diff2'].dropna())
print('Second order ADF Statistic:', adf_result_diff2[0])
print('Second order p-value:', adf_result_diff2[1])

# Plot ACF and PACF after second order difference
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(data['Temperature change_diff2'].dropna(), ax=plt.gca(), lags=40)
plt.title('Autocorrelation Function (Second Order Difference)')
plt.subplot(122)
plot_pacf(data['Temperature change_diff2'].dropna(), ax=plt.gca(), lags=40)
plt.title('Partial Autocorrelation Function (Second Order Difference)')
plt.savefig('pics/acf_pacf_diff2.png', dpi=350)
plt.close()

# Model fitting
model = sm.tsa.statespace.SARIMAX(data['Temperature change'], order=(7, 1, 2), trend='c')
model = model.fit(disp=-1)
print("Order: (7, 1, 2)")

# AIC criterion
print("AIC:", model.aic)

# Model Diagnostic
# In sample prediction
prediction = model.get_prediction()
prediction_mean = prediction.predicted_mean

# Calculate remainder sequence
remainder = data['Temperature change'] - prediction_mean

# Plot the remainder sequence
plt.figure(figsize=(10, 6))
plt.plot(data.index, remainder, marker='o')
plt.title('Remainder Sequence')
plt.xlabel('Year')
plt.ylabel('Remainder')
plt.grid(True)
plt.savefig('pics/remainder_sequence.png', dpi=350)
plt.close()

# Plot the ACF And PACF of the remainder sequence
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(remainder.dropna(), ax=plt.gca(), lags=40)
plt.title('Autocorrelation Function of Remainder Sequence')
plt.subplot(122)
plot_pacf(remainder.dropna(), ax=plt.gca(), lags=40)
plt.title('Partial Autocorrelation Function of Remainder Sequence')
plt.savefig('pics/acf_pacf_remainder.png', dpi=350)
plt.close()

# Ljung-Box test of remainder sequence
ljung_box_result_remainder = acorr_ljungbox(remainder.dropna(), lags=[1], return_df=True)
print("Ljung-Box Test Result of Remainder Sequence:")
print(ljung_box_result_remainder)

# Calculate MSE
mse = mean_squared_error(data['Temperature change'], prediction_mean)
print("Mean Squared Error:", mse)

# Out of sample prediction
forecast = model.get_forecast(steps=10)

# Plot the prediction and forecast
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Temperature change'], label='Original')
plt.plot(data.index, prediction_mean, label='Prediction')
plt.plot(range(data.index[-1]+1, data.index[-1]+11), forecast.predicted_mean, label='Forecast')
plt.title('Prediction and Forecast')
plt.xlabel('Year')
plt.ylabel('Temperature Change')
plt.legend()
plt.grid(True)
plt.savefig('pics/prediction_forecast.png', dpi=350)
plt.close()

# # Save test results to output.csv
# test_results = {
#     'Original Ljung-Box Test Result': ljung_box_result,
#     'First Order Ljung-Box Test Result': ljung_box_result_diff,
#     'Second Order Ljung-Box Test Result': ljung_box_result_diff2,
#     'Remainder Ljung-Box Test Result': ljung_box_result_remainder,
#     'Original ADF p-value': [adf_result[1]],
#     'First Order ADF p-value': [adf_result_diff[1]],
#     'Second Order ADF p-value': [adf_result_diff2[1]],
#     'Mean Squared Error': [mse],
#     'AIC': [model.aic]
# }

# Save test results to output.csv
test_results = {
    'Original Ljung-Box p-value': ljung_box_result['lb_pvalue'].values,
    'First Order Ljung-Box p-value': ljung_box_result_diff['lb_pvalue'].values,
    'Second Order Ljung-Box p-value': ljung_box_result_diff2['lb_pvalue'].values,
    'Remainder Ljung-Box p-value': ljung_box_result_remainder['lb_pvalue'].values,
    'Original ADF p-value': [adf_result[1]],
    'First Order ADF p-value': [adf_result_diff[1]],
    'Second Order ADF p-value': [adf_result_diff2[1]],
    'Mean Squared Error': [mse],
    'AIC': [model.aic]
}
test_results_df = pd.DataFrame(test_results)
test_results_df.to_csv('output.csv', index=False)