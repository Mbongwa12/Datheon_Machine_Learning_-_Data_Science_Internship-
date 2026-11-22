# Multi-Class Benchmark Classification Pipeline — Iris

End-to-end supervised classification pipeline built with Scikit-Learn, benchmarking
**Logistic Regression** against a **Decision Tree** on the Iris dataset with leakage-free feature
scaling, full metric reporting, confusion matrices, one-vs-rest ROC curves and decision-boundary
visualization.

**Datheon Machine Learning & Data Science Internship — Task Submission**
Author: Mbongwa (Andile) · Data Science III, Sol Plaatje University

---

## Results

| Metric | Logistic Regression | Decision Tree |
|---|---|---|
| Test accuracy (30 held-out samples) | 0.9333 (28/30) | **0.9667 (29/30)** |
| Precision (macro) | 0.9333 | **0.9697** |
| Recall (macro) | 0.9333 | **0.9667** |
| F1 (macro) | 0.9333 | **0.9666** |
| ROC AUC (one-vs-rest, macro) | **0.9968** | 0.9718 |
| 5-fold CV accuracy | 0.9533 ± 0.0452 | **0.9600 ± 0.0249** |
| Misclassifications | 2 | 1 |

Per-class ROC AUC:

| Class | Logistic Regression | Decision Tree |
|---|---|---|
| setosa | 1.000 | 1.000 |
| versicolor | 0.995 | 0.945 |
| virginica | 0.995 | 0.970 |

### What the numbers mean

- The two models are separated by **one test sample**, which on a 30-sample test set is worth 3.3
  percentage points. Cross-validation narrows the gap to 0.960 vs 0.953 — inside the fold-to-fold
  standard deviation, so the models are effectively tied.
- **Every error either model makes is _versicolor_ vs _virginica_.** `setosa` is never missed and
  scores AUC = 1.000 for both. Both models got the same borderline flower wrong — a *versicolor*
  with unusually large petals.
- **Logistic regression signals its uncertainty; the tree does not.** Its two errors carried
  confidences of 0.515 and 0.557 (effectively coin flips, catchable by a review threshold). The
  tree's single error carried confidence 1.000. That coarse probability resolution is why the
  tree's ROC curve is step-shaped and its AUC trails despite the higher accuracy.
- **Petal length and petal width carry the signal.** The tree assigns them essentially all of its
  Gini importance; logistic regression puts its largest coefficients on the same pair. Sepal width
  contributes almost nothing.

---

## Architecture

```
            data/iris.csv  (or the raw GitHub URL)
                      |
                      v
        +-------------------------------+
        |  Audit: dtypes, nulls, dupes, |
        |  class balance (50/50/50)     |
        +-------------------------------+
                      |
                      v
        +-------------------------------+
        |  EDA: pairplot, correlation   |
        |  heatmap, per-class boxplots  |
        +-------------------------------+
                      |
                      v
        +-------------------------------+
        |  Stratified 80/20 split       |
        |  (120 train / 30 test, rs=42) |
        +-------------------------------+
                      |
          +-----------+-----------+
          v                       v
 +------------------+   +------------------+
 | Pipeline         |   | Pipeline         |
 | StandardScaler   |   | StandardScaler   |
 |   -> LogisticReg |   |   -> DecisionTree|
 |      max_iter=1k |   |      max_depth=3 |
 +------------------+   +------------------+
          |                       |
          +-----------+-----------+
                      v
        +-------------------------------+
        |  Evaluation                   |
        |  - classification_report      |
        |  - confusion matrices (raw +  |
        |    row-normalised)            |
        |  - OvR ROC + micro/macro AUC  |
        |  - decision boundaries (2D)   |
        |  - coefficients / importances |
        |  - 5-fold stratified CV       |
        +-------------------------------+
                      |
                      v
            figures/*.png   results/*.csv
```

### Design decisions

**The scaler lives inside a `Pipeline`, not before the split.** `StandardScaler` must learn its mean
and standard deviation from training data only. Fitting it on the full dataset before splitting
leaks test statistics into training and biases every reported score upward. Section 6 of the
notebook demonstrates the absence of leakage directly: after transformation the training set has
mean 0.000 / std 1.000 while the test set does not — proof the scaler never saw it. The pipeline
form also means the scaler is refitted on each fold during cross-validation, which a manual
pre-split scaling cannot do.

**The tree keeps the scaler even though it does not need one.** A decision tree splits on thresholds
of raw feature values and is therefore scale-invariant. The scaler is retained so both models travel
an identical preprocessing path, which keeps the benchmark fair and the two pipelines
interchangeable at the call site.

**`max_depth=3` is fixed a priori.** An unconstrained tree on 120 training samples memorises them
and reports a misleading 100% training accuracy. Depth 3 is enough for three classes separated on
two features, and keeps the printed tree readable as an explanation artefact.

**The split is stratified.** Without `stratify=y`, a 30-row test set can easily end up class-skewed,
making per-class recall unstable. Stratification preserves the 1/3 – 1/3 – 1/3 proportions in both
parts.

---

## Repository layout

```
.
├── iris_classification_pipeline.ipynb   # main deliverable, executed with outputs embedded
├── data/
│   └── iris.csv                         # committed copy so the notebook reruns offline
├── figures/                             # all 11 charts, exported as PNG
│   ├── 01_class_balance.png
│   ├── 02_pairplot.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_feature_boxplots.png
│   ├── 05_metric_benchmark.png
│   ├── 06_confusion_matrices.png
│   ├── 07_confusion_matrices_normalised.png
│   ├── 08_roc_curves.png
│   ├── 09_decision_boundaries.png
│   ├── 10_decision_tree.png
│   └── 11_feature_importance.png
├── results/                             # machine-readable metrics
│   ├── benchmark_summary.csv
│   ├── classification_report_logistic_regression.csv
│   ├── classification_report_decision_tree.csv
│   ├── roc_auc_table.csv
│   ├── cross_validation_summary.csv
│   └── final_results.csv
├── requirements.txt
└── README.md
```

---

## Installation

Requires Python 3.10 or newer.

```bash
# 1. Clone
git clone https://github.com/Mbongwa12/Datheon_Machine_Learning_Data_Science_Internship.git
cd Datheon_Machine_Learning_Data_Science_Internship/task-1-iris-classification

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter lab iris_classification_pipeline.ipynb
```

Then **Run → Restart Kernel and Run All Cells**. Every result is seeded with `random_state=42`, so a
rerun reproduces the numbers in this README exactly.

### Reproducing without a browser

```bash
pip install nbconvert ipykernel
jupyter nbconvert --to notebook --execute --inplace iris_classification_pipeline.ipynb
```

The notebook reads the dataset from the public raw URL and falls back to `data/iris.csv` when
offline, so it runs either way.

---

## Dataset

Iris — 150 samples, 4 numeric features (sepal length, sepal width, petal length, petal width in cm),
3 balanced classes (*setosa*, *versicolor*, *virginica*, 50 each).

Source: `https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv`

---

## Limitations and next steps

Iris is a near-solved benchmark; high scores here confirm the pipeline is correct, not that the
problem is hard. With only 30 test samples the single-split difference between the two models is not
statistically meaningful.

Next steps:

1. `GridSearchCV` over tree depth and the logistic-regression penalty / `C`, nested inside the
   cross-validation loop so the tuning itself is not leaked.
2. Repeat the identical pipeline on the **Wine** dataset (13 features, mildly imbalanced), where
   feature scaling and macro-averaging carry real weight.
3. Add a calibration curve for the tree to quantify the overconfidence shown in Section 9.
