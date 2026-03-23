# Text as Data – Homework 3  
**L1-Regularized Logistic Regression**

This repository contains the code and write-up for Homework 3 of the *Text as Data* course.  
The assignment compares L2 (ridge) and L1 (lasso) regularized logistic regression for binary text classification using TF-IDF features.

## Files

- `hw03_l2_baseline.py`  
  Baseline script reproducing the Week 09 tutorial pipeline: TF-IDF vectorization → logistic regression with L2 penalty → evaluation.

- `hw03_l1_model.py`  
  Modified pipeline using L1 regularization (`penalty="l1"`, `solver="liblinear"`), with the same data and TF-IDF settings for a fair comparison.

## Data

Both scripts expect the following JSON files produced in the Week 09 tutorial:

- `train_core_vs_neg.json`
- `test_core_vs_neg.json`

These files are not included in this repository. Place them in the same directory as the scripts before running.

## Methods

The analysis proceeds in four main stages:

1. Load pre-split train/test JSON data from the Week 09 tutorial
2. Vectorize texts using TF-IDF (`min_df=5`, `max_df=0.95`)
3. Train logistic regression under both L2 and L1 penalties
4. Evaluate and compare: confusion matrix, classification report, ROC AUC, sparsity diagnostics, and top positive/negative-weight words visualized as separate horizontal bar charts

## Output

Each script saves results to a dedicated folder under `output/`:

- `output/l2/`
  - `metrics.txt` — confusion matrix, classification report, ROC AUC, sparsity
  - `top15_positive_words.png` — bar chart of top 15 CORE-predictive words
  - `top15_negative_words.png` — bar chart of top 15 NEG-predictive words

- `output/l1/`
  - `metrics.txt` — confusion matrix, classification report, ROC AUC, sparsity
  - `top15_positive_words.png` — bar chart of top 15 CORE-predictive words
  - `negative_words.png` — bar chart of negative-weight words (only non-zero)

The interpretive comparison write-up is submitted separately on Canvas.

---
