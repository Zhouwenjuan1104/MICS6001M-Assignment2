import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 设定随机种子，确保结果可复现
np.random.seed(42)
n = 1000  # 采样点数

# 定义 MA(q) 模型参数
models = [
    {"label": r"MA(1): $x_t = \epsilon_t - 2\epsilon_{t-1}$", "ma": [1, -2]},  # MA(1)
    {"label": r"MA(2): $x_t = \epsilon_t - 0.5\epsilon_{t-1}$", "ma": [1, -0.5]},  # MA(2)
    {"label": r"MA(3): $x_t = \epsilon_t - \frac{4}{5}\epsilon_{t-1} + \frac{16}{25}\epsilon_{t-2}$",
     "ma": [1, -4 / 5, 16 / 25]},  # MA(3)
    {"label": r"MA(4): $x_t = \epsilon_t - \frac{5}{4}\epsilon_{t-1} + 2\frac{25}{16}\epsilon_{t-2}$",
     "ma": [1, -5 / 4, 25 / 16]}  # MA(4)
]

# 创建子图
fig, axes = plt.subplots(4, 5, figsize=(20, 16))

# 遍历所有模型
for i, model in enumerate(models):
    # 生成时间序列
    ma = model["ma"]
    process = ArmaProcess(ar=[1], ma=ma)
    ts = process.generate_sample(nsample=n)

    # 1. **时间序列折线图**
    axes[i, 0].plot(ts, color='blue')
    axes[i, 0].set_title(f"Time Series: {model['label']}")

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
plt.savefig('pics/task3_time_series.png', dpi=350)
plt.show()

# ============= 判断可逆性并打印完整信息 =============
print("\n========= MA Process Invertibility Check =========")
for idx, model in enumerate(models, 1):
    ma = model["ma"]
    arma_process = sm.tsa.ArmaProcess(ar=[1], ma=ma)

    # 获取特征方程的根并计算模长
    roots = arma_process.maroots
    moduli = np.abs(roots)

    # 判断可逆性
    is_invertible_by_roots = all(mod > 1.0 for mod in moduli)  # 所有根模长 >1
    is_invertible_by_statsmodels = arma_process.isinvertible  # 库函数判断

    # 格式化根和模长输出
    roots_str = [f"{root.real:.2f}" if np.isclose(root.imag, 0) else f"{root:.2f}" for root in roots]
    moduli_str = [f"{mod:.2f}" for mod in moduli]

    # 输出结果
    print(f"Model {idx} MA Polynomial Roots: {roots_str}")
    print(f"Model {idx} Root Moduli: {moduli_str}")
    print(f"Model {idx} Roots Inside Unit Circle: {not is_invertible_by_roots}")
    print(f"Model {idx} Invertible (by roots): {is_invertible_by_roots}")
    print(f"Model {idx} Invertible (by statsmodels): {is_invertible_by_statsmodels}\n")

# ============= ACF 和 PACF 图 =============

fig, axes = plt.subplots(4, 2, figsize=(12, 16))

for i, model in enumerate(models):
    ma = model["ma"]
    process = ArmaProcess(ar=[1], ma=ma)
    ts = process.generate_sample(nsample=n)

    # 4. **ACF 自相关图**
    plot_acf(ts, lags=20, ax=axes[i, 0])
    axes[i, 0].set_title(f"ACF: {model['label']}")

    # 5. **PACF 偏自相关图**
    plot_pacf(ts, lags=20, ax=axes[i, 1])
    axes[i, 1].set_title(f"PACF: {model['label']}")

# 调整布局并保存
plt.tight_layout()
plt.savefig('pics/task3_acf_pacf.png', dpi=350)
plt.show()
