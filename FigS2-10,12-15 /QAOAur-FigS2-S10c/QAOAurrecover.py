import os
import numpy as np
import boot  # 必须在同级目录下

# ============================================================
# 辅助函数：普通最小二乘法 (OLS) 拟合
# ============================================================
def fit_ab_ols(n_fit, success_fit):
    """
    拟合模型: success = -log(p) = log(a) + n * log(b)
    对应原方程: 1/p = a * b^n
    """
    n_fit = np.asarray(n_fit, float)
    y_fit = np.asarray(success_fit, float)  # y = -log(p)

    # 线性回归: y = beta0 + beta1 * x
    X = np.column_stack([np.ones_like(n_fit), n_fit])
    beta, *_ = np.linalg.lstsq(X, y_fit, rcond=None)
    
    loga = float(beta[0])
    logb = float(beta[1])
    
    a = float(np.exp(loga))
    b = float(np.exp(logb))

    # 计算 R2
    y_pred = loga + logb * n_fit
    sse = float(np.sum((y_fit - y_pred) ** 2))
    sst = float(np.sum((y_fit - float(np.mean(y_fit))) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    
    return a, b, loga, logb, r2

# ============================================================
# 核心处理函数
# ============================================================
def recover_multi_try(
    npz_paths,
    out_path,
    n_start=5,           # 数据起始 n
    fit_n_min=9,         # 拟合起始 n
    fit_n_max=None,      # 拟合结束 n (None 表示取最大值)
    ci=0.95,
    B_point=2000,
    B_param=2000,
    eps=1e-12
):
    # 1. 加载数据
    success_list = []
    loaded_paths = []
    
    for p in npz_paths:
        if not os.path.exists(p):
            print(f"[Warning] File not found: {p}")
            continue
        try:
            d = np.load(p, allow_pickle=True)
            # 假设 npz 里存的是 'success' (-log p)
            s = np.asarray(d['success'], float)
            success_list.append(s[:7])
            loaded_paths.append(p)
        except Exception as e:
            print(f"[Error] Failed to load {p}: {e}")

    if not success_list:
        raise RuntimeError("No valid data loaded.")

    # 2. 对齐数据长度 (取交集)
    min_len = min(len(s) for s in success_list)
    success_matrix = np.stack([s[:min_len] for s in success_list], axis=0) # Shape: (R, N_total)
    
    # 构造 n 列表
    n_list = np.arange(n_start, n_start + min_len, dtype=int)
    
    # 转换回概率 p = exp(-success)
    # Shape: (R, N_total)
    p_runs = np.clip(np.exp(-success_matrix), eps, 1.0)
    R, N_total = p_runs.shape

    # 3. 计算均值曲线
    p_mean = np.clip(p_runs.mean(axis=0), eps, 1.0)
    
    # 4. 点态 Bootstrap CI (调用 boot.py)
    # 对每一个 n 点做 bootstrap
    p_lo = np.zeros(N_total)
    p_hi = np.zeros(N_total)
    
    for i in range(N_total):
        # 传入该 n 处的所有 try 数据
        lo, hi = boot.mean_boot(p_runs[:, i], ci=ci, B=B_point)
        p_lo[i] = lo
        p_hi[i] = hi
    
    # 5. 准备拟合数据
    # 确定拟合范围
    actual_max = n_list[-1]
    limit_max = fit_n_max if fit_n_max is not None else actual_max
    
    mask = (n_list >= fit_n_min) & (n_list <= limit_max)
    
    if np.sum(mask) < 3:
        raise ValueError(f"Valid points for fitting < 3. Range: {fit_n_min}-{limit_max}, Data: {n_list[0]}-{n_list[-1]}")

    n_fit = n_list[mask]
    p_mean_fit = p_mean[mask]
    success_mean_fit = -np.log(p_mean_fit) # 用于 OLS
    
    # 6. 计算点估计 (a_hat, b_hat)
    a_hat, b_hat, loga_hat, logb_hat, r2_hat = fit_ab_ols(n_fit, success_mean_fit)
    
    # 7. 参数 Bootstrap CI (调用 boot.py)
    # boot.fit_boot 拟合的是 p ~ a_dec * b_dec^n (衰减)
    # 我们需要的 scaling 是 1/p ~ a * b^n (增长)
    # 关系: a = 1/a_dec, b = 1/b_dec
    
    # p_runs_fit shape: (R, N_fit) -> 转置为 (N_fit, R) 传给 boot.fit_boot
    p_runs_fit = p_runs[:, mask]
    
    # 获取衰减参数的 CI
    a_dec_ci, b_dec_ci = boot.fit_boot(n_fit, p_runs_fit.T, ci=ci, B=B_param, eps=eps)
    
    # 翻转区间得到增长参数 CI: [1/hi, 1/lo]
    a_ci = np.array([1.0 / a_dec_ci[1], 1.0 / a_dec_ci[0]])
    b_ci = np.array([1.0 / b_dec_ci[1], 1.0 / b_dec_ci[0]])

    # 8. 转换输出数据 (success = -log p, y = 1/p)
    success_mean = -np.log(p_mean)
    success_ci_lo = -np.log(np.clip(p_hi, eps, 1.0))
    success_ci_hi = -np.log(np.clip(p_lo, eps, 1.0))

    y_mean = 1.0 / p_mean
    y_ci_lo = 1.0 / np.clip(p_hi, eps, 1.0)
    y_ci_hi = 1.0 / np.clip(p_lo, eps, 1.0)

    # 9. 保存
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    
    np.savez_compressed(
        out_path,
        loaded_paths=np.array(loaded_paths, dtype=object),
        n_list=n_list,
        n_start=n_start,
        fit_n_min=fit_n_min,
        fit_n_max=limit_max,
        R=R,
        ci=ci,
        
        # 原始概率统计
        p_mean=p_mean,
        p_ci_lo=p_lo,
        p_ci_hi=p_hi,
        
        # 转换后统计
        success_mean=success_mean,
        y_mean=y_mean,
        y_ci_lo=y_ci_lo,
        y_ci_hi=y_ci_hi,
        
        # 拟合结果
        a_hat=a_hat,
        b_hat=b_hat,
        loga_hat=loga_hat,
        logb_hat=logb_hat,
        r2_hat=r2_hat,
        
        # 拟合 CI
        a_ci=a_ci,
        b_ci=b_ci
    )

    print(f"[{os.path.basename(out_path)}] Fitted n=[{fit_n_min}, {limit_max}] (Points={len(n_fit)})")
    print(f"  b_hat = {b_hat:.6f} (CI: {b_ci[0]:.6f}, {b_ci[1]:.6f})")
    print(f"  R2    = {r2_hat:.6f}")


# ============================================================
# 主程序调用示例
# ============================================================
if __name__ == "__main__":
    # 配置部分
    PREFIX = "QAOAurdata/QAOAur"
    OUT_DIR = "QAOAurrecover"
    K_VALUES = [0.55, 0.575, 0.6, 0.626, 0.65, 0.675, 0.7, 0.725, 0.75]
    
    # 循环处理每个 k
    for k in K_VALUES:
        # 1. 构造文件名列表 (try0, try1, try2)
        npz_list = [
            f"{PREFIX}{k}try0.npz",
            f"{PREFIX}{k}try1.npz",
            f"{PREFIX}{k}try2.npz"
        ]
        
        # 2. 构造输出路径
        out_path = os.path.join(OUT_DIR, f"recoverQAOAur{k}.npz")
        
        # 3. 执行恢复与拟合
        # 只要数据允许，fit_n_max 默认为 None 会一直拟合到最后
        recover_multi_try(
            npz_list,
            out_path,
            n_start=5,
            fit_n_min=5,      # 要求的最小拟合范围
            fit_n_max=11,   # 不设上限，自动使用最大可用 n
            ci=0.95
        )