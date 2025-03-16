import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 设置随机种子
np.random.seed(42)
n = 1000  # 生成的数据点数

# 定义 ARMA 模型
models = {
    "AR(1)": ArmaProcess(ar=[1, -0.8], ma=[1]),
    "MA(1)": ArmaProcess(ar=[1], ma=[1, 0.7]),
    "ARMA(1,1) (0.8, 0.7)": ArmaProcess(ar=[1, -0.8], ma=[1, 0.7]),
    "ARMA(1,1) (-0.8, -0.7)": ArmaProcess(ar=[1, 0.8], ma=[1, -0.7])
}

# 创建子图
fig, axes = plt.subplots(4, 5, figsize=(20, 16))

timeseries_data = {}

for i, (label, process) in enumerate(models.items()):
    ts = process.generate_sample(nsample=n)
    timeseries_data[label] = ts

    # 画时间序列图
    axes[i, 0].plot(ts, color='blue')
    axes[i, 0].set_title(f"Time Series: {label}")

    # 直方图 & 密度图
    sns.histplot(ts, bins=30, kde=True, ax=axes[i, 1], color='purple')
    axes[i, 1].set_title("Histogram & Density")

    # 箱线图
    sns.boxplot(x=ts, ax=axes[i, 2], color='green')
    axes[i, 2].set_title("Boxplot")

    # Lag-1 和 Lag-2 散点图
    axes[i, 3].scatter(ts[:-1], ts[1:], alpha=0.5, color='red')
    axes[i, 3].set_title("Lag-1 Plot")

    axes[i, 4].scatter(ts[:-2], ts[2:], alpha=0.5, color='orange')
    axes[i, 4].set_title("Lag-2 Plot")

plt.tight_layout()
plt.savefig('pics/task4_time_series.png', dpi=350)
plt.show()

# **检查平稳性 & 可逆性**
for label, process in models.items():
    print(f"\n====== {label} ======")
    print(f"{label} is Stationary (ArmaProcess): {process.isstationary}")
    print(f"{label} is Invertible (ArmaProcess): {process.isinvertible}")

    # **ADF 检验**
    ts = timeseries_data[label]
    adf_result = adfuller(ts)
    print(f"ADF Statistic = {adf_result[0]:.4f}, p-value = {adf_result[1]:.4f}")

    if adf_result[1] < 0.05:
        print(f"{label} is stationary (Reject H0)")
    else:
        print(f"{label} is NOT stationary (Fail to Reject H0)")

# **绘制 ACF & PACF 图**
fig, axes = plt.subplots(4, 2, figsize=(12, 16))

for i, (label, ts) in enumerate(timeseries_data.items()):
    plot_acf(ts, lags=20, ax=axes[i, 0])
    axes[i, 0].set_title(f"ACF: {label}")

    plot_pacf(ts, lags=20, ax=axes[i, 1])
    axes[i, 1].set_title(f"PACF: {label}")

plt.tight_layout()
plt.savefig('pics/task4_acf_pacf.png', dpi=350)
plt.show()
