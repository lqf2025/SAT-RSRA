# 1.System Requirements

## Software Dependencies

This project requires **Python 3.12.7** and the following Python libraries:

- **NumPy** (>= 1.24)
- **SciPy** (>= 1.10)
- **Matplotlib** (>= 3.7)
- **Parfor** (>= 1.0)
- **PySAT** (>= 0.9)
- **JAX** (>= 0.4)
- **Random** (standard Python library, no installation required)
- **Math** (standard Python library, no installation required)

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```txt
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
parfor>=1.0
jax>=0.4
python-sat>=0.9
```

Alternatively, you can install them manually:

```bash
pip install numpy>=1.24 scipy>=1.10 matplotlib>=3.7 parfor>=1.0 python-sat>=0.9
```

# 2. Installation Guide

## Step 1: Clone the Repository

To begin, clone the repository to your local machine. Open a terminal or command prompt and run the following command:

```bash
git clone https://github.com/lqf2025/SAT-RSRA.git
cd SAT-RSRA
```

## Step 2: Set Up a Virtual Environment (Recommended)

It is highly recommended to use a virtual environment to manage the project dependencies and avoid conflicts with other Python packages on your system.

#### For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Once activated, you should see (venv) in your terminal or command prompt, indicating that you’re working within the virtual environment.

## Step 3: Install Dependencies

Now, install the required Python dependencies using pip. This will ensure that all necessary packages are installed for the software to run correctly.

```bash
pip install -r requirements.txt
```


## Step 4: Verify Installation

After installing the dependencies, you can verify that all required packages have been installed correctly by running the following command:

```bash
pip list
```

This will display a list of installed packages and their versions.
You should see the following (or similar) packages in the list:

```txt
numpy (version 1.24+)
scipy (version 1.10+)
matplotlib (version 3.7+)
parfor (version 1.0+)
jax (version 0.4+)
python-sat (version 0.9+)
```

# 3. Project Structure and Usage

## Project Organization

The project folder is organized into seven main sections, each corresponding to the generation of specific figures for the main text or supplementary information.

The data within these folders is categorized into three levels:

1.  **Raw Data**
    * Comprises full simulation logs, commonly named after XXXdata.
    * *Note: These are computationally expensive to generate.*

2.  **Processed Data**
    * Contains estimated averages and scaling factors derived via bootstrapping, commonly named after RecoverXXXX.
    * *Processing time:* Seconds to minutes (based on raw data).

3.  **Visualization Scripts**
    * Scripts for generating final figures and tables.
    * Allows for immediate visualization using the processed data.

- Each folder contains a Python script with the same name as the folder. Running the script directly produces the corresponding figures using pre-computed data.
- The remaining folders contain the evaluation codes and data required for generating these figures, appending the corresponding image location as a suffix. 
- `generator.py` is used to generate random positive one-in-three SAT instances, which occurs multiple times. After generation, it applies our reduction algorithm to produce the matrix **L**.
- `generators.py` is used to generate random  one-in-three SAT instances with mixed polarity, which occurs multiple times. After generation, it applies our reduction algorithm to produce the matrix **L**.
- `boot.py` provides bootstrap resampling used to estimate confidence intervals for sample means and exponential fit parameters ($y≈a\cdot b^n$).
- Because many of the calculations are extremely time-consuming and repetitive, several scripts must be run multiple times with varying parameters (e.g., different m/n ratios and numbers of repetitions).  We recommend implementing the codes on a server in **parallel mode**.

---

## Figure Descriptions and Related Files

### **Fig2 – Classical Simulations**

Covers:
- Reduction ability
- Feasibility of the enhanced quantum solvers
- Determination of the necessary number of layers for enahnced QAA- and QAOA-based solvers

**Scripts and outputs:**
- `reduction.py` (folder `reduction-Fig2abh`) 
→ evaluates the reduction ability of the RARA, with output  `reduction.npz`.
- `QAA.py` and `QAOA.py` (folder `QAAQAOA-Fig2de`) 
→ simulate enhanced QAA- and QAOA-based solvers for case stored in `PQC100,63.npz`, with outputs `QAAdraw.npz` and  `QAOAdraw1.npz`-`QAOAdraw16.npz`.
You can adjust the trange2/t variable (line 140 in QAA.py and line 124 in QAOA.py), which controls the maximum number of layers for both solvers.  The `QAOAdraw1.npz`-`QAOAdraw16.npz` are further comined by  `QAOAcombine.py` to form `QAOAdraw_combined.npz`.

   ⭐These scripts serve as demos for the two solvers. In our setup, we set trange2 in QAA.py to 12 and in QAOA.py to 1, allowing the demo to run on a personal laptop in minutes. For generating the full dataset used in the main text, set trange2 in QAA.py to 52 and in QAOA.py to 1-16.
- `VQE.py` and `VQEcount.py` (folder `VQE-Fig2c`)  → simulate enhanced VQE-based solver for case stored in `PQC150,94.npz`.
  - `VQE.py` → generates successful and failed attempts, with output `VQE.npz`. 

   ⭐This script also serves as a demo for the VQE-based solver. According to the estimation in the main text, the VQE-based solver has an 8.3\% success probability, meaning it is expected to find a solution within approximately 10 minutes.
  - `VQEcount.py` → calculates the probability of success, with output `VQEcount0.npz`-`VQEcount5.npz`.
- `QAAp.py` and `QAOAp.py` (folder `pdeterminate-Fig2fg`) → evaluate scaling performance of enhanced QAA- and QAOA-based solvers under varying numbers of layers.  
  - `QAAp.py` outputs: `QAAtry0.npz – QAAtry4.npz`, each containing 2500 samples, forming a total dataset of 10000 samples.  Bootstrapped using `QAAprecover.py` to form `RecoverQAAp.npz`.
  - `QAOAp.py` outputs: results stored in folder `QAOAsingle`.  ⚠️ **Note:** `QAOAp.py` can be extremely time-consuming.

---

### **Fig4 – First Experiment**

Summarizes results from the first experiment, divided into four classes of solvers.

**Data files:**
- `QAOA-Fig4cg` → enhanced QAOA-based solver
- `QAOAur-Fig4dh` → original QAOA-based solver
- `VQE-Fig4ae` → enhanced VQE-based solver
- `VQEur-Fig4bf` → original VQE-based solver

- Unnumbered `.npz` (e.g., `QAOA.npz`) → accurate results from classical simulations.
- Numbered `.npz` (e.g., `QAOA1.npz – QAOA5.npz`) → experimental results.

---

### **Fig5 – Second Experiment**

Summarizes results from the second experiment.

- Generated cases: The case folder is divided into two sections: `original`, containing cases for $n<=20$, and `revise`, containing cases for $n=20$ and $n=21$.
- Original data: `exptotal` folder  
- Summary: `Exp2.xlsx`

---

### **FigS2–S10,12-15 – Exponential Scaling for positive one-in-three**

Reports results related to exponential scaling under varying m/n ratios.

**Data for table:**
-`datauni.py` for scaling fitted in the largest range.
-`dataeq.py` for scaling fitted between 25 and 45.

**Scripts and outputs:**
- `QAAsingle.py` (folder `QAA-FigS2-S10b`) → scalings of enhanced QAA-based solver, with outputs in the form `QAAsinglesXX.npz` in subfolder `QAAdata`. Bootstrap using `recoverQAA.py' for 'recoverQAAXX.npz' in subfolder `QAArecoverwhole'. If fit data between 25 and 45 is wanted, instead bootstrap using `recoverQAAeq.py' for 'recoverQAAeqXX.npz' in subfolder  `QAArecovereq'.
- `QAAur.py` (folder `QAAur-FigS2-S10b`) → scalings of original QAA-based solver, with outputs in the form `QAAurXX.npz` in subfolder `QAAurdata`. Bootstrap using `QAAurrecover` for 'recoverQAAurXX.npz` in subfolder  `QAAurrecover'. 
- `QAOA.py` (folder `QAOA-FigS2-S10c`) → scalings of enhanced QAOA-based solver, with outputs in the form `QAOAsXXtryX.npz` in subfolder `QAOAdata`.  Bootstrap using `recoverQAOA.py` for `recoverQAOAsXX.npz` in subfolder `QAOArecover`. 
- `QAOAur.py` (folder `QAOAur-FigS2-S10c`) → scalings of original QAOA-based solver, with outputs in the form `QAOAurXXtryX.npz` in subfolder `QAOAurdata`. Bootstrap using `QAOAurrecover.py` for `recoverQAOAurXX.npz` in subfolder `QAOAurrecover`. 
- `uniclassical.py` (folder `classical-FigS2-S10a`) → scalings of classical solvers and enhanced VQE-based solver, with outputs in the form `uniXXpX.npz` in folder `unidata`. Bootstrap using `recoverwhole` for 'Classical_kXX.npz' in subfolder `unirecoverwhole`.  If fit between 25 and 45 is wanted, instead bootstrap using `recovereq` for 'Classical_keqXX.npz' in subfolder `unirecovereq`. 

---
### **FigS11 – Exponential Scaling for RSRA-aware baselines**

Reports results related to exponential scaling of RSRA-baseline under varying m/n ratios.

**Data for table:**
-`databaseline.py` for scaling fits.

**Scripts and outputs:**
- `Flip.py` (folder `Flip-FigS11b`) → scaling of Flip-RSRA baseline, with outputs in the form `FlipXX_raw.npz` in subfolder `Flipdata`. Bootstrap using `Fliprecover.py` for a total `RecoverFilpall.npz`.
- `RG.py` (folder `RG-FigS11acd`) → scaling of Sampling-RSRA, Grover-RSRA, sampling-2SAT baselines, with outputs in the form `RGXX.npz` in subfolder `RGdata`. Bootstrap using `RGrecover.py` for `recover_RGXX.npz` in subfolder `RGrecover`.

---
### **FigS16-18 – Exponential Scaling for one-in-three with mixed polarity**

Reports results related to exponential scaling under varying m/n ratios and mixed polarity.

**Data for table:**
-`datamix.py` for scaling fits.

**Scripts and outputs:**
- `uniclassicalmix.py` (folder `Classical-FigS16-18a`) → scaling of classical solvers, with outputs in the form `uniclassicalmixXX.npz`, bootstraped using `recoverunimix` for `RecoverXXuniclassical.npz`
- `QAAmix.py` (folder `QAA-Classical-FigS16-18b`) → scaling of enhanced QAA-based solver with outputs in the form `QAAmixXX.npz`, bootstraped using `QAAmixrecover` for `QAAmixrecoverXX.npz`.
- `QAOAmix.py` (folder `QAOA-Classical-FigS16-18c`) → scaling of enhanced QAOA-based solver, with outputs in the form `QAOAmix_kXX_tryX.npz`, bootstraped using `recoverQAOAmix` for `RecoverQAOAmixXX.npz`.



---

### **FigS20 – Barren Plateau Verification**

Numerical verification of the absence of barren plateau.

**Scripts and outputs:**
- `BP.py` (folder `BP`) → calculates average variance under a fixed m/n ratio and various problem sizes, with outputs in the form `BPXX.npz`. 



## License

This code is released under the MIT License. See the `LICENSE` file for details.


