# TrialOptimizer


---

This repository contains the code for the method described above.

## 1. System requirements

### 1.1 Software requirements
- Python: [3.11.0]
- Required packages:
    - lifelines==0.30.0          
    - matplotlib==3.8.0                   
    - numpy==1.26.4                     
    - pandas==2.2.3     
    - pytorch==2.5.1
    - pymoo==0.6.1.5             
    - scikit-learn==1.2.2            
    - scipy==1.11.4          
    - statsmodels==0.14.0

### 1.2 Operating systems
The software has been tested on:
- macOS (Apple silicon / Intel)
The code is expected to be compatible with Linux-based systems.

### 1.3 Hardware requirements
- CPU: Standard desktop computer
- GPU: Optional, not required

---

## 2. Installation guide

Clone the repository:

bash
git clone https://github.com/Penncil/TrialOptimizer.git
cd TrialOptimizer

Approximately 1–3 minutes on a standard desktop computer.

---

## 3. Demo

### 3.1 Instructions to Run
Launch Jupyter Notebook; Open analysis.ipynb; Run all cells.

### 3.2 Expected Output
- The notebook executes without errors
- Intermediate results (e.g., training progress, figures) are displayed inline
- Final outputs are generated within the notebook 

The demo uses simulated data and is intended only to verify that the code executes correctly and is consistent with the method. It does not reproduce the main results in the manuscript.

### 3.3 Expected run time for demo
- Approximately <1 minutes on a standard laptop for one eligibility configuration.

---

## 4. Instructions for Use

### 4.1 Running on Your Own Data
To apply the method to your own dataset, prepare your data in the following format:
- CSV file
- Columns:
  - covaraites
  - outcome
  - treatment variable
  - multiple negative control outcomes
  - eligibility criteria (both fixed and potentially relaxed)
- Modify the data loading section in the notebook: df = load_data("your_data.csv")
- Replace the column names in code by your data column names

### 4.2 Reproducibility (Optional)
Reproducing the main results in the manuscript requires access to the full dataset described in the paper.
The provided simulated data is intended for demonstration purposes only.

