import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# 设定随机种子，确保结果可复现
np.random.seed(42)
n = 1000  # 采样点数

# 定义AR模型参数
models = [
    {"label": r"AR(1): $x_t = 0.8x_{t-1} + \epsilon_t$", "ar": [1, -0.8]},  # AR(1) 系数 0.8
    {"label": r"AR(2): $x_t = -0.8x_{t-1} + \epsilon_t$", "ar": [1, 0.8]},  # AR(2) 系数 -0.8
    {"label": r"AR(3): $x_t = x_{t-1} - 0.5x_{t-2} + \epsilon_t$", "ar": [1, -1, 0.5]},  # AR(3)
    {"label": r"AR(4): $x_t = -x_{t-1} - 0.5x_{t-2} + \epsilon_t$", "ar": [1, 1, 0.5]}  # AR(4)
]

# 创建子图
fig, axes = plt.subplots(4, 5, figsize=(20, 16))

# 遍历所有模型
for i, model in enumerate(models):
    # 生成时间序列
    ar = model["ar"]
    process = ArmaProcess(ar=ar, ma=[1])
    ts = process.generate_sample(nsample=n)

    # 1. **时间序列折线图**
    axes[i, 0].plot(ts, color='blue')
    axes[i, 0].set_title(f"Time Series: {model['label']}")

    # 平稳性检测
    adf_result = adfuller(ts)
    is_stationary = "Yes" if adf_result[1] < 0.05 else "No"
    print(f"{model['label']} - Stationary: {is_stationary}, ADF p-value: {adf_result[1]:.4f}")

    # 2. **直方图, 密度图, 盒须图**
    sns.histplot(ts, bins=30, kde=True, ax=axes[i, 1], color='purple')
    axes[i, 1].set_title("Histogram & Density")

    sns.boxplot(x=ts, ax=axes[i, 2], color='green')
    axes[i, 2].set_title("Boxplot")

    # 3. **滞后图（Lag-1 & Lag-2）**
    axes[i, 3].scatter(ts[:-1], ts[1:], alpha=0.5, color='red')
    axes[i, 3].set_title("Lag-1 Plot")

    axes[i, 4].scatter(ts[:-2], ts[2:], alpha=0.5, color='orange')
    axes[i, 4].set_title("Lag-2 Plot")

# 调整布局
plt.tight_layout()
plt.savefig('pics/task2_time_series.png', dpi=350)
plt.show()

# ============= ACF 和 PACF 图 =============

fig, axes = plt.subplots(4, 2, figsize=(12, 16))

for i, model in enumerate(models):
    ar = model["ar"]
    process = ArmaProcess(ar=ar, ma=[1])
    ts = process.generate_sample(nsample=n)

    # 4. **ACF 自相关图**
    plot_acf(ts, lags=20, ax=axes[i, 0])
    axes[i, 0].set_title(f"ACF: {model['label']}")

    # 5. **PACF 偏自相关图**
    plot_pacf(ts, lags=20, ax=axes[i, 1])
    axes[i, 1].set_title(f"PACF: {model['label']}")

# 调整布局并保存
plt.tight_layout()
plt.savefig('pics/task2_acf_pacf.png', dpi=350)
plt.show()
