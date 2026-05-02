import numpy as np
import matplotlib.pyplot as plt
from boot import fit_boot, mean_boot

# 1. 整理数据
for k in [0.3,0.5,0.07]:
    data_file = f"QAAmix{k}.npz"
    # 加载原始数据，并按 (n_points, trials) 整理
    # 形状 (55, 10000, 1) -> (55, 10000)
    data = np.load(data_file)['single2_all'].reshape(55, 10000)
    ns = np.arange(5, 60)
    success_raw = data 

    # 2. 计算 p 的统计量 (均值及其 Bootstrap 置信区间)
    p_means, p_lows, p_highs = [], [], []
    print("正在计算成功概率 p 的 Bootstrap 置信区间...")
    for i in range(len(ns)):
        m = np.mean(success_raw[i])
        # 计算均值的 95% Bootstrap CI
        lo, hi = mean_boot(success_raw[i, :], ci=0.95, B=2000)
        p_means.append(m)
        p_lows.append(lo)
        p_highs.append(hi)

    p_means = np.array(p_means)
    p_lows = np.array(p_lows)
    p_highs = np.array(p_highs)

    # 3. 转换为复杂度指标 1/p
    # 1/p 的估计值为 1/mean(p)
    # 1/p 的置信区间为 [1/p_high, 1/p_low] (倒数会翻转顺序)
    eps = 1e-15
    inv_means = 1.0 / np.clip(p_means, eps, None)
    inv_lows = 1.0 / np.clip(p_highs, eps, None)
    inv_highs = 1.0 / np.clip(p_lows, eps, None)

    # 4. 执行拟合分析 (针对 n >= 45，即索引 40 以后)
    print("正在对 1/p 缩放趋势执行拟合分析...")
    # 首先利用 boot.py 拟合成功概率: p = a * b^n
    a_ci_succ, b_ci_succ = fit_boot(ns[40:], success_raw[40:], ci=0.95, B=1000)

    # 计算成功概率的点估计系数
    x_fit = ns[40:]
    y_fit_log = np.log(np.clip(p_means[40:], eps, None))
    coeffs_p = np.polyfit(x_fit, y_fit_log, 1)

    b_hat_succ = np.exp(coeffs_p[0])
    a_hat_succ = np.exp(coeffs_p[1])

    # --- 新增：计算 R2 (针对对数空间下的线性拟合) ---
    y_predicted = np.polyval(coeffs_p, x_fit)
    ss_res = np.sum((y_fit_log - y_predicted)**2)
    ss_tot = np.sum((y_fit_log - np.mean(y_fit_log))**2)
    r2 = 1 - (ss_res / ss_tot)

    # 转换为复杂度指标 1/p = A * B^n 的参数
    B_hat = 1.0 / b_hat_succ
    A_hat = 1.0 / a_hat_succ

    # 映射 B 的 95% 置信区间
    B_ci = (1.0 / b_ci_succ[1], 1.0 / b_ci_succ[0])
    A_ci = (1.0 / a_ci_succ[1], 1.0 / a_ci_succ[0])

    # 6. 保存绘图所需的参数到 npz (增加 r2)
    np.savez(
        f"QAAmixrecover{k}.npz",
        ns=ns,
        inv_means=inv_means,
        inv_lows=inv_lows,
        inv_highs=inv_highs,
        A_hat=A_hat,
        B_hat=B_hat,
        A_ci=A_ci,
        B_ci=B_ci,
        b_hat_succ=b_hat_succ,
        b_ci_succ=b_ci_succ,
        r2=r2  # 这里保存了 r2
    )

    print(f"✅ 拟合复杂度基数 B: {B_hat:.6f}")
    print(f"✅ R^2 拟合优度: {r2:.6f}")
    print(f"✅ 处理后的绘图参数已保存至: QAAmixrecover{k}.npz")