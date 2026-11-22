# Datheon Machine Learning & Data Science Internship

Portfolio of completed tasks for the Datheon Machine Learning & Data Science internship
(4-week programme, November 2026).

**Author:** Mbongwa (Andile) · Data Science III, Sol Plaatje University

---

## Tasks in this repository

| # | Task | Deliverable | Status |
|---|---|---|---|
| 1 | Multi-Class Benchmark Classification Pipeline (Iris) | [`task-1-iris-classification/`](task-1-iris-classification/) | ✅ Complete |

Each task folder is self-contained: notebook, data, exported figures, machine-readable results and
its own README explaining the method and findings.

---

## Task 1 — Multi-Class Benchmark Classification Pipeline (Iris)

End-to-end supervised classification with Scikit-Learn: leakage-free feature scaling, a stratified
80/20 split, and a head-to-head benchmark of **Logistic Regression** against a **Decision Tree**.

**Notebook:** [`task-1-iris-classification/iris_classification_pipeline.ipynb`](task-1-iris-classification/iris_classification_pipeline.ipynb)

| Metric | Logistic Regression | Decision Tree |
|---|---|---|
| Test accuracy (30 held-out samples) | 0.9333 (28/30) | **0.9667 (29/30)** |
| F1 (macro) | 0.9333 | **0.9666** |
| ROC AUC (one-vs-rest, macro) | **0.9968** | 0.9718 |
| 5-fold CV accuracy | 0.9533 ± 0.0452 | **0.9600 ± 0.0249** |

**Key finding.** The tree wins on accuracy by exactly one test sample but loses decisively on AUC.
Logistic regression's two errors carried confidences of 0.515 and 0.557 — it effectively flagged
them as coin flips. The tree's single error carried confidence **1.000**: a depth-3 leaf that was
pure in training reports total certainty on a flower it got wrong. Which model to ship therefore
depends on whether the output needs a trustworthy confidence score or a human-readable rule.

Every misclassification either model makes is *versicolor* vs *virginica*; *setosa* is never missed
by either (AUC = 1.000).

Full method, design decisions and limitations: [`task-1-iris-classification/README.md`](task-1-iris-classification/README.md)

---

## Repository structure

```
.
├── README.md                          # this file — index of all tasks
├── requirements.txt                   # shared dependencies, pinned
├── .gitignore
└── task-1-iris-classification/
    ├── README.md                      # task-specific write-up
    ├── iris_classification_pipeline.ipynb
    ├── data/iris.csv                  # committed so the notebook reruns offline
    ├── figures/                       # 11 exported charts (PNG)
    └── results/                       # metrics as CSV
```

---

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Mbongwa12/Datheon_Machine_Learning_Data_Science_Internship.git
cd Datheon_Machine_Learning_Data_Science_Internship

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

jupyter lab
```

Open any task notebook and run **Kernel → Restart Kernel and Run All Cells**. Every result is seeded
with `random_state=42`, so a rerun reproduces the numbers above exactly.

---

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · JupyterLab
