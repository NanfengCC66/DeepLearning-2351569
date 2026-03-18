import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

os.environ.pop("QT_DEVICE_PIXEL_RATIO", None)
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

# ============================================================
# 0. 超参数配置
# ============================================================
class HyperParameters:
    """集中管理所有超参数，方便调整实验"""
    x_min: float = -3.0          # 输入域左端点
    x_max: float = 3.0           # 输入域右端点
    train_size: int = 800        # 训练集样本数
    test_size: int = 200         # 测试集样本数
    noise_std: float = 0.05      # 训练集噪声标准差（模拟真实采样误差）
    hidden1: int = 64            # 第一隐藏层神经元数
    hidden2: int = 64            # 第二隐藏层神经元数
    epochs: int = 5000           # 训练轮次
    lr: float = 1e-3             # Adam 学习率
    beta1: float = 0.9           # Adam 一阶矩衰减系数
    beta2: float = 0.999         # Adam 二阶矩衰减系数
    eps: float = 1e-8            # Adam 数值稳定项
    seed: int = 42               # 随机种子，保证可复现

hp = HyperParameters()


# ============================================================
# 1. 目标函数定义
# ============================================================
def target_function(x: np.ndarray) -> np.ndarray:
    """
    目标函数: f(x) = sin(2π x) · exp(-0.5x) + 0.3x²

    参数
    ----
    x : shape (N,) 或 (N,1) 的 ndarray

    返回
    ----
    y : 与 x 同形状的 ndarray
    """
    return np.sin(2 * np.pi * x) * np.exp(-0.5 * x) + 0.3 * x ** 2


# ============================================================
# 2. 数据采集
# ============================================================
def generate_data(hp: HyperParameters):
    """
    生成训练集和测试集

    训练集: 均匀采样 + 高斯噪声（模拟真实观测误差）
    测试集: 均匀采样（无噪声，用于客观评估）

    返回
    ----
    x_train, y_train, x_test, y_test : 均为 shape (N,1) 的 ndarray
    """
    rng = np.random.default_rng(hp.seed)

    # 训练集
    x_train = rng.uniform(hp.x_min, hp.x_max, size=(hp.train_size, 1))
    noise    = rng.normal(0, hp.noise_std, size=(hp.train_size, 1))
    y_train  = target_function(x_train) + noise

    # 测试集（均匀排列，便于绘图）
    x_test = np.linspace(hp.x_min, hp.x_max, hp.test_size).reshape(-1, 1)
    y_test = target_function(x_test)

    return x_train, y_train, x_test, y_test


# ============================================================
# 3. ReLU 神经网络（纯 NumPy 手动实现）
# ============================================================

# ---------- 激活函数 ----------
def relu(z: np.ndarray) -> np.ndarray:
    """ReLU: max(0, z)"""
    return np.maximum(0.0, z)

def relu_grad(z: np.ndarray) -> np.ndarray:
    """ReLU 导数: 1 if z>0 else 0"""
    return (z > 0).astype(np.float64)


# ---------- 参数初始化 ----------
def init_params(hp: HyperParameters) -> dict:
    """
    He 初始化（专为 ReLU 设计）:
        W ~ N(0, sqrt(2/fan_in))

    网络结构: 1 -> hidden1 -> hidden2 -> 1
    """
    rng = np.random.default_rng(hp.seed)

    def he(fan_in, fan_out):
        return rng.normal(0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))

    params = {
        'W1': he(1,          hp.hidden1),   # (1,   64)
        'b1': np.zeros((1,   hp.hidden1)),  # (1,   64)
        'W2': he(hp.hidden1, hp.hidden2),   # (64,  64)
        'b2': np.zeros((1,   hp.hidden2)),  # (1,   64)
        'W3': he(hp.hidden2, 1),            # (64,  1 )
        'b3': np.zeros((1,   1)),           # (1,   1 )
    }
    return params


# ---------- 前向传播 ----------
def forward(x: np.ndarray, params: dict) -> tuple:
    """
    前向传播

    参数
    ----
    x      : (N, 1)
    params : 网络参数字典

    返回
    ----
    y_pred : (N, 1)  网络输出
    cache  : 中间变量，供反向传播使用
    """
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    W3, b3 = params['W3'], params['b3']

    z1 = x  @ W1 + b1          # (N, hidden1)
    a1 = relu(z1)               # (N, hidden1)

    z2 = a1 @ W2 + b2          # (N, hidden2)
    a2 = relu(z2)               # (N, hidden2)

    z3 = a2 @ W3 + b3          # (N, 1)
    y_pred = z3                 # 输出层不加激活（线性回归）

    cache = (x, z1, a1, z2, a2)
    return y_pred, cache


# ---------- 损失函数 ----------
def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """均方误差: (1/N) * Σ(y_pred - y_true)²"""
    return float(np.mean((y_pred - y_true) ** 2))


# ---------- 反向传播 ----------
def backward(y_pred: np.ndarray, y_true: np.ndarray,
             params: dict, cache: tuple) -> dict:
    """
    反向传播（链式法则手动推导）

    返回
    ----
    grads : 与 params 同键的梯度字典
    """
    x, z1, a1, z2, a2 = cache
    N = x.shape[0]

    W2, W3 = params['W2'], params['W3']

    # 输出层梯度
    dL_dz3 = 2.0 * (y_pred - y_true) / N   # (N, 1)

    dW3 = a2.T @ dL_dz3                    # (hidden2, 1)
    db3 = dL_dz3.sum(axis=0, keepdims=True)# (1, 1)

    # 第二隐藏层梯度
    dL_da2 = dL_dz3 @ W3.T                 # (N, hidden2)
    dL_dz2 = dL_da2 * relu_grad(z2)        # (N, hidden2)

    dW2 = a1.T @ dL_dz2                    # (hidden1, hidden2)
    db2 = dL_dz2.sum(axis=0, keepdims=True)# (1, hidden2)

    # 第一隐藏层梯度
    dL_da1 = dL_dz2 @ W2.T                 # (N, hidden1)
    dL_dz1 = dL_da1 * relu_grad(z1)        # (N, hidden1)

    dW1 = x.T @ dL_dz1                     # (1, hidden1)
    db1 = dL_dz1.sum(axis=0, keepdims=True)# (1, hidden1)

    grads = {
        'W1': dW1, 'b1': db1,
        'W2': dW2, 'b2': db2,
        'W3': dW3, 'b3': db3,
    }
    return grads


# ---------- Adam 优化器 ----------
class AdamOptimizer:
    """
    Adam 优化器
    参考: Kingma & Ba, 2015, "Adam: A Method for Stochastic Optimization"

    update 规则:
        m = β1*m + (1-β1)*g
        v = β2*v + (1-β2)*g²
        m̂ = m / (1-β1^t)
        v̂ = v / (1-β2^t)
        θ = θ - lr * m̂ / (√v̂ + ε)
    """
    def __init__(self, params: dict, hp: HyperParameters):
        self.lr    = hp.lr
        self.beta1 = hp.beta1
        self.beta2 = hp.beta2
        self.eps   = hp.eps
        self.t     = 0  # 时间步

        # 初始化一阶矩和二阶矩（与参数同形状，全零）
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict) -> dict:
        """执行一步 Adam 更新，返回更新后的参数"""
        self.t += 1
        t = self.t
        new_params = {}

        for k in params:
            g = grads[k]

            # 更新矩估计
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g ** 2

            # 偏差修正
            m_hat = self.m[k] / (1 - self.beta1 ** t)
            v_hat = self.v[k] / (1 - self.beta2 ** t)

            # 参数更新
            new_params[k] = params[k] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return new_params


# ============================================================
# 4. 训练流程
# ============================================================
def train(x_train: np.ndarray, y_train: np.ndarray,
          hp: HyperParameters) -> tuple:
    """
    训练 ReLU 网络

    返回
    ----
    params     : 训练完毕的参数字典
    loss_history : 每轮 MSE 损失列表
    """
    params   = init_params(hp)
    optimizer = AdamOptimizer(params, hp)
    loss_history = []

    print(f"{'Epoch':>8} | {'Train MSE':>12}")
    print("-" * 24)

    for epoch in range(1, hp.epochs + 1):
        # 前向
        y_pred, cache = forward(x_train, params)

        # 损失
        loss = mse_loss(y_pred, y_train)
        loss_history.append(loss)

        # 反向
        grads = backward(y_pred, y_train, params, cache)

        # 参数更新
        params = optimizer.step(params, grads)

        # 每 500 轮打印一次
        if epoch % 500 == 0 or epoch == 1:
            print(f"{epoch:>8} | {loss:>12.6f}")

    return params, loss_history


# ============================================================
# 5. 评估指标
# ============================================================
def evaluate(x_test: np.ndarray, y_test: np.ndarray,
             params: dict) -> dict:
    """
    在测试集上评估模型

    返回指标
    --------
    mse  : 均方误差
    rmse : 均方根误差
    mae  : 平均绝对误差
    r2   : 决定系数 R²
    """
    y_pred, _ = forward(x_test, params)

    mse  = float(np.mean((y_pred - y_test) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(y_pred - y_test)))
    ss_res = float(np.sum((y_pred - y_test) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2   = 1.0 - ss_res / ss_tot

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}


# ============================================================
# 6. 可视化
# ============================================================
def plot_results(x_train, y_train, x_test, y_test,
                 params, loss_history, hp):
    """绘制三张图：损失曲线、拟合对比、残差分布"""

    y_pred_test, _ = forward(x_test, params)

    # --- 连续真实曲线（用于对比）---
    x_dense = np.linspace(hp.x_min, hp.x_max, 1000).reshape(-1, 1)
    y_dense_true = target_function(x_dense)
    y_dense_pred, _ = forward(x_dense, params)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("ReLU 神经网络函数逼近实验结果", fontsize=15, fontweight='bold')

    # ---- 图1: 训练损失曲线 ----
    ax = axes[0]
    ax.semilogy(loss_history, color='steelblue', linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("训练损失曲线")
    ax.grid(True, alpha=0.3)
    ax.annotate(f"最终 MSE = {loss_history[-1]:.6f}",
                xy=(len(loss_history)-1, loss_history[-1]),
                xytext=(0.45, 0.75), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='red'),
                color='red', fontsize=10)

    # ---- 图2: 拟合对比 ----
    ax = axes[1]
    ax.scatter(x_train, y_train, s=8, alpha=0.3, color='gray',
               label='训练数据（含噪声）')
    ax.plot(x_dense, y_dense_true,  color='dodgerblue',
            linewidth=2.5, label='真实函数 f(x)')
    ax.plot(x_dense, y_dense_pred, color='orangered',
            linewidth=2.0, linestyle='--', label=r'网络预测 $\hat{f}(x)$')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("函数拟合对比")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- 图3: 残差分布 ----
    ax = axes[2]
    residuals = (y_pred_test - y_test).flatten()
    ax.hist(residuals, bins=30, color='mediumpurple',
            edgecolor='white', alpha=0.85)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel("残差 (预测值 - 真实值)")
    ax.set_ylabel("频数")
    ax.set_title("测试集残差分布")
    ax.grid(True, alpha=0.3)
    ax.text(0.03, 0.92,
            f"μ={residuals.mean():.4f}\nσ={residuals.std():.4f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig("relu_approximation_result.png", dpi=150, bbox_inches='tight')
    print("\n可视化结果已保存至 relu_approximation_result.png")
    plt.show()


# ============================================================
# 7. 主程序入口
# ============================================================
def main():
    print("=" * 60)
    print("   基于 ReLU 神经网络的函数逼近实验")
    print("=" * 60)
    print(f"\n目标函数: f(x) = sin(2πx)·exp(-0.5x) + 0.3x²")
    print(f"输入域  : [{hp.x_min}, {hp.x_max}]")
    print(f"网络结构: 1 → {hp.hidden1}(ReLU) → {hp.hidden2}(ReLU) → 1")
    print(f"训练集  : {hp.train_size} 样本（噪声 σ={hp.noise_std}）")
    print(f"测试集  : {hp.test_size} 样本（无噪声）")
    print(f"训练轮次: {hp.epochs}，学习率: {hp.lr}\n")

    # Step 1: 数据生成
    x_train, y_train, x_test, y_test = generate_data(hp)
    print(f"数据生成完毕 | 训练集: {x_train.shape} | 测试集: {x_test.shape}\n")

    # Step 2: 训练
    params, loss_history = train(x_train, y_train, hp)

    # Step 3: 测试集评估
    metrics = evaluate(x_test, y_test, params)
    print("\n" + "=" * 40)
    print("      测试集评估结果")
    print("=" * 40)
    for name, val in metrics.items():
        print(f"  {name:<6}: {val:.6f}")
    print("=" * 40)

    # Step 4: 可视化
    plot_results(x_train, y_train, x_test, y_test,
                 params, loss_history, hp)


if __name__ == "__main__":
    main()