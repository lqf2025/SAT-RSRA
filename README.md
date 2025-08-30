# 1.System Requirements

## Software Dependencies

This project requires **Python 3.12.7** and the following Python libraries:

- **NumPy** (>= 1.24)
- **SciPy** (>= 1.10)
- **Matplotlib** (>= 3.7)
- **Parfor** (>= 1.0)
- **PySAT** (>= 0.9)
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
pysat>=0.9
```

Alternatively, you can install them manually:

```bash
pip install numpy>=1.24 scipy>=1.10 matplotlib>=3.7 parfor>=1.0 pysat>=0.9
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
python -m venv Venn
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
pysat (version 0.9+)
```

# 3. Project Structure and Usage

The project folder is organized into five main parts, each corresponding to the generation of one or more figures in the main text or supplementary information.

- Each folder contains a Python script with the same name as the folder. Running the script directly produces the corresponding figures using pre-computed data.
- The remaining folders contain the evaluation codes and data required for generating these figures, appending the corresponding image location as a suffix. 
- `generator.py` is used to generate random positive 3-SAT instances, which occurs multiple times. After generation, it applies our reduction algorithm to produce the matrix **L**.
- Because many of the calculations are extremely time-consuming and repetitive, several scripts must be run multiple times with varying parameters (e.g., different m/n ratios and numbers of repetitions).  We recommend implementing the codes on a server in **parallel mode**.

---

## Figure Descriptions and Related Files

### **Fig2 – Classical Simulations**

Covers:
- Reduction ability
- Feasibility of enhanced solvers
- Determination of the necessary number of layers

**Scripts and outputs:**
- `reductionability.py` (folder `reduction-Fig2abh`) 
→ evaluates the reduction ability of the RARA, with output  `reduction.npz`.
- `QAA.py` and `QAOA.py` (folder `QAAQAOA-Fig2de`) 
→ simulate enhanced QAA- and QAOA-based solvers for case stored in `PQC100,63.npz`, with output `QAAdraw.npz` and  `QAOAdraw.npz`.
- `VQE.py` and `VQEcount.py` (folder `VQE-Fig2c`)  → simulate enhanced VQE-based solver for case stored in `PQC150,94.npz`.
  - `VQE.py` → generates successful and failed attempts, with output `VQE.npz`.  
  - `VQEcount.py` → calculates the probability of success, with output `VQEcount0.npz`-`VQEcount5.npz`.
- `QAAp.py` and `QAOAp.py` (folder `pdeterminate-Fig2fg`) → evaluate scaling performance of enhanced QAA- and QAOA-based solvers under varying numbers of layers.  
  - `QAAp.py` outputs: `QAAscale0.npz – QAAscale4.npz`  
  - `QAOAp.py` outputs: results stored in folder `QAOAsingle`.  
  ⚠️ **Note:** `QAOAp.py` can be extremely time-consuming.

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

- Generated cases: `case` folder  
- Original data: `originaldata` folder  
- Summary: `Exp2.xlsx`

---

### **FigS2–S10 – Exponential Scaling**

Reports results related to exponential scaling under varying m/n ratios.

**Scripts and outputs:**
- `QAAsingle.py` (folder `QAA-FigS2-S10b`) → scalings of enhanced QAA-based solver, with outputs in the form `QAAsinglellXX.npz`.
- `QAAur.py` (folder `QAAur-FigS2-S10b`) → scalings of original QAA-based solver, with outputs in the form `QAAscaleurXX.npz`.
- `QAOAsingle.py` (folder `QAOA-FigS2-S10c`) → scalings of enhanced QAOA-based solver, with outputs in the form `QAOAsingleXX.npz`.
- `QAOAur.py` (folder `QAOAur-FigS2-S10c`) → scalings of original QAOA-based solver, with outputs in the form `QAOAurscaleXX.npz`.
- `uniclassical.py` (folder `classical-FigS2-S10a`) → scalings of classical solvers and enhanced VQE-based solver, with outputs in the form `uniXX.npz`.

---

### **FigS12 – Barren Plateau Verification**

Numerical verification of the absence of barren plateau.

**Scripts and outputs:**
- `BP.py` (folder `BP`) → calculates average variance under a fixed m/n ratio and various problem sizes, with outputs in the form `BPXX.npz`.


