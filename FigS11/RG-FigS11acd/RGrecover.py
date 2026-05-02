import numpy as np
from boot import mean_boot, fit_boot
from scipy.stats import linregress

def collect_and_process(k, n_min=10, n_max=20):
    # Loads raw data, computes statistical metrics (Direct RSRA, TwoSAT, Grover) with confidence intervals, and saves the results.
    
    n_list = np.arange(n_min, n_max, dtype=int)

    # Load data from file
    data = np.load(f"RGdata/RG{k}.npz", allow_pickle=True)
    dim_samples = data["dim_samples"]
    count_samples = data["count_samples"]
    count_samples2 = data["count_samples2"]

    # First metric: 2^dim / samples (Inverse probability)
    directRSRA = (2 ** dim_samples) / count_samples

    # Second metric: samples2 / samples (Ratio of 2-SAT solutions)
    twoSAT = count_samples2 / count_samples

    # Third metric: sqrt(2^dim) (Grover search complexity)
    grover = (np.sqrt(2) ** dim_samples)

    directRSRA_ci = np.array([mean_boot(directRSRA[ni,:], ci=0.95, B=2000) for ni in range(len(n_list))])
    twoSAT_ci = np.array([mean_boot(twoSAT[ni,:], ci=0.95, B=2000) for ni in range(len(n_list))])
    grover_ci = np.array([mean_boot(grover[ni,:], ci=0.95, B=2000) for ni in range(len(n_list))])

    directRSRA_ci_lo = directRSRA_ci[:, 0]
    directRSRA_ci_hi = directRSRA_ci[:, 1]
    directRSRA_ci_ave = np.sqrt(directRSRA_ci_lo * directRSRA_ci_hi)

    twoSAT_ci_lo = twoSAT_ci[:, 0]
    twoSAT_ci_hi = twoSAT_ci[:, 1]
    twoSAT_ci_ave = np.sqrt(twoSAT_ci_lo * twoSAT_ci_hi)

    grover_ci_lo = grover_ci[:, 0]
    grover_ci_hi = grover_ci[:, 1]
    grover_ci_ave = np.sqrt(grover_ci_lo * grover_ci_hi)

    directRSRAa_ci, directRSRAb_ci = fit_boot(n_list, directRSRA, ci=0.95, B=2000)
    twoSATa_ci, twoSATb_ci = fit_boot(n_list, twoSAT, ci=0.95, B=2000)
    grovera_ci, groverb_ci = fit_boot(n_list, grover, ci=0.95, B=2000)

    slope, intercept, r_value, p_value, std_err = linregress(n_list, np.log(directRSRA_ci_ave))  
    directRSRAr2 = r_value ** 2  

    slope, intercept, r_value, p_value, std_err = linregress(n_list, np.log(twoSAT_ci_ave))  
    twoSAT_r2 = r_value ** 2  

    slope, intercept, r_value, p_value, std_err = linregress(n_list, np.log(grover_ci_ave))  
    grover_r2 = r_value ** 2  

    # Save results to file
    out = f"RGrecover/recover_RG{k}.npz"
    np.savez_compressed(
        out,
        k=float(k),
        n_list=n_list,
        
        # directRSRA data
        directRSRA_ci_lo=directRSRA_ci_lo,
        directRSRA_ci_hi=directRSRA_ci_hi,
        directRSRA_ci_ave=directRSRA_ci_ave,
        directRSRAr2=directRSRAr2,
        directRSRAa_ci=directRSRAa_ci,
        directRSRAb_ci=directRSRAb_ci,
        
        # grover data
        grover_ci_lo=grover_ci_lo,
        grover_ci_hi=grover_ci_hi,
        grover_ci_ave=grover_ci_ave,
        grover_r2=grover_r2,
        grovera_ci=grovera_ci,
        groverb_ci=groverb_ci,

        # twoSAT data
        twoSAT_ci_lo=twoSAT_ci_lo,
        twoSAT_ci_hi=twoSAT_ci_hi,
        twoSAT_ci_ave=twoSAT_ci_ave,
        twoSAT_r2=twoSAT_r2,
        twoSATa_ci=twoSATa_ci,
        twoSATb_ci=twoSATb_ci
    )
    print(out)

# Call function
for k in [0.55, 0.575, 0.6, 0.626, 0.65, 0.675, 0.7, 0.725, 0.75]:
    collect_and_process(k=k, n_min=10, n_max=20)