import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import adfuller

# 生成 AR 模型的时间序列
np.random.seed(42)
n = 1000  # 采样点数

# AR 模型参数
models = [
    {"ar": [1, -0.8], "ma": [1]},  # AR(1): 0.8x_{t-1}
    {"ar": [1, 1.1], "ma": [1]},  # AR(2): -1.1x_{t-1}
    {"ar": [1, -1, 0.5], "ma": [1]},  # AR(3): x_{t-1} - 0.5x_{t-2}
    {"ar": [1, -1, -0.5], "ma": [1]}  # AR(4): x_{t-1} + 0.5x_{t-2}
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# 生成序列并检查平稳性
print("\n========= Stationarity Check =========\n")
for i, model in enumerate(models):
    ar = model["ar"]
    ma = model["ma"]

    # 使用 statsmodels 创建 ARMA 过程实例
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    ts = arma_process.generate_sample(nsample=n)

    # 绘制时间序列图
    axes[i].plot(ts)
    axes[i].set_title(f"AR Model {i + 1}")

    #通过 ArmaProcess 的 stationary 检查
    print(f"Model {i + 1} Stationary (ArmaProcess):", arma_process.isstationary)

    #计算单位根 (Unit Root)
    roots = np.roots(ar)
    moduli = np.abs(roots)
    print(f"Model {i + 1} AR Polynomial Roots: {roots}")
    print(f"Model {i + 1} Root Moduli: {moduli}")
    print(f"Model {i + 1} Roots Inside Unit Circle: {np.all(moduli < 1)}")  # 平稳性要求所有根的模大于 1

    #*DF 检验
    adf_result = adfuller(ts)
    p_value = adf_result[1]
    print(f"Model {i + 1} ADF p-value: {p_value:.4f}")

    # ADF 结果解释
    if p_value < 0.05:
        print(f"Model {i + 1} is Stationary (Reject H0)")
    else:
        print(f"Model {i + 1} is NOT Stationary (Fail to Reject H0)")

    print("=" * 60)

# 调整子图间距并保存
plt.tight_layout()
plt.savefig('pics/task1_line_plot.png', dpi=350)
plt.show()
