1. System Requirements

Software Dependencies:


To run this, you will need Python 3.12.7 and the following Python libraries:

NumPy (version 1.24+)

SciPy (version 1.10+)

Matplotlib (version 3.7+)

Parfor (version 1.0+)

PySAT (version 0.9+)

Random (standard Python library, no installation required)

Math (standard Python library, no installation required)


You can install all dependencies using the following command:

pip install -r requirements.txt

Where requirements.txt contains:
numpy>=1.24

scipy>=1.10

matplotlib>=3.7

parfor>=1.0

pysat>=0.9


Operating Systems:

The software is compatible with the following operating systems:

Windows: 10 and 11 (64-bit)

macOS: 10.15+ (64-bit)

Linux: Ubuntu 20.04+ (64-bit) or other distributions with Python 3.12.7 or higher.


Versions the Software Has Been Tested On:

The software has been tested on the following versions:

Python: 3.12.7

NumPy (version 1.26.4)

SciPy (version 1.11.4)

Matplotlib (version  3.10.1)

Parfor (version 2025.1.0)

python-sat  (version 1.8.dev14)


While other versions of these libraries may also work, these are the versions the software has been explicitly tested with.

2. Installation Guide

Step 1: Clone the Repository

To begin, clone the repository to your local machine. Open a terminal or command prompt and run the following command:

git clone https://github.com/lqf2025/SAT-RSRA.git

cd SAT-RSRA

Step 2: Set Up a Virtual Environment (Recommended)

It is highly recommended to use a virtual environment to manage the project dependencies and avoid conflicts with other Python packages on your system.

For macOS/Linux:

python3 -m venv venv

source venv/bin/activate

For Windows:

python -m venv venv

venv\Scripts\activate

Once activated, you should see (venv) in your terminal or command prompt, indicating that you’re working within the virtual environment.

Step 3: Install Dependencies

Now, install the required Python dependencies using pip. This will ensure that all necessary packages are installed for the software to run correctly.

pip install -r requirements.txt

Step 4: Verify Installation

After installing the dependencies, you can verify that all required packages have been installed correctly by running the following command:

pip list

This will display a list of installed packages and their versions. You should see the following (or similar) packages in the list:

numpy (version 1.24+)

scipy (version 1.10+)

matplotlib (version 3.7+)

parfor (version 1.0+)

pysat (version 0.9+)

openpyxl (version 3.0+)

Typical Install Time on a Normal Desktop Computer

On a typical desktop computer , and a stable internet connection, the installation should take approximately 10–15 minutes:

Cloning the repository: ~1-2 minutes.

Setting up the virtual environment: ~1 minute.

Installing dependencies: ~5-10 minutes, depending on internet speed and the number of packages.


This means the total time to complete the setup should be around 10–15 minutes on a "normal" desktop computer.

3. Project Structure and Usage

The project folder is organized into five main parts, each corresponding to the generation of one or more figures in the main text or supplementary information.

Each folder contains a Python script with the same name as the folder. Running the script directly produces the corresponding figures using pre-computed data.

The remaining folders contain the evaluation codes and data required for generating these figures, appending the corresponding image location as a suffix. 

generator.py, is used to generate random positive 3-SAT instances, which occurs multiple times. After generation, it applies our reduction algorithm to produce the matrix L.

Because many of the calculations are extremely time-consuming and repetitive, several scripts need to be run multiple times with varying parameters (e.g., different m/n ratios and numbers of repetitions).


Figure Descriptions and Related Files


Fig2 – Classical Simulations

Fig2 presents results related to classical simulations, covering:

Reduction ability

Feasibility of enhanced solvers

Determination of the necessary number of layers

Scripts and outputs:

reductionability.py (folder reduction-Fig2abh)→ generates reduction.npz in the reduction folder.

QAA.py and QAOA.py (folder QAAQAOA-Fig2de) → simulate QAA- and QAOA-based solvers for case stored in PQC100,63.npz.

VQE.py and VQEcount.py (folder VQE-Fig2c) → simulate VQE-based solvers for case stored in PQC150,94.npz.

VQE.py generates successful and failed attempts.

VQEcount.py calculates the probability of success.

QAAp.py and QAOAp.py (folder pdeterminate-Fig2fg) → evaluate scaling performance under varying numbers of layers.

QAAp.py outputs: QAAscale0.npz – QAAscale4.npz

QAOAp.py outputs: results stored in the folder QAOAsingle.

!!QAOAp.py can be extremely time-consuming.

Fig4 – First Experiment

Fig4 summarizes results from the first experiment, divided into four classes of solvers.

QAOA-Fig4cg contains results related with enhanced QAOA-based solver.

QAOAur-Fig4dh contains results related with original QAOA-based solver.

VQE-Fig4ae contains results related with enhanced VQE-based solver.

VQEur-Fig4bf contains results related with original VQE-based solver.

Unnumbered .npz files (e.g., QAOA.npz) → accurate results obtained from classical simulations.

Numbered .npz files (e.g., QAOA1.npz – QAOA5.npz) → experimental results.

Fig5 – Second Experiment

Fig5 summarizes results from the second experiment.

Generated cases: case folder

Original data: originaldata folder

Summary: Exp2.xlsx


FigS2–S10 – Exponential Scaling

These figures report results related to exponential scaling.

Scripts and locations:

QAAsingle.py (folder QAA-FigS2-S10b) → data for enhanced QAA-based solver

QAAur.py (folder QAAur-FigS2-S10b) → data for original QAA-based solver

QAOAsingle.py (folder QAOA-FigS2-S10c) → data for enhanced QAA-based solver

QAOAur.py (folder QAOAur-FigS2-S10c) → data for original QAOA-based solver

uniclassical.py (folder classical-FigS2-S10a) → data for classical solvers and  enhanced VQE-based solver.

FigS12 – Barren Plateau Verification

Figure S12 presents numerical verification of barren plateau phenomena.

Script BP.py (folder BP)→ generates average variance under a fixed m/n ratio and various problem sizes.










