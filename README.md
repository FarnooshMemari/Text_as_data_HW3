Text as Data – Homework 3
L1-Regularized Logistic Regression
This repository contains the code and write-up for Homework 3 of the Text as Data course.  
The assignment compares L2 (ridge) and L1 (lasso) regularized logistic regression for binary text classification using TF-IDF features.
Files
`hw03_l2_baseline.py`  
Baseline script reproducing the Week 09 tutorial pipeline: TF-IDF vectorization → logistic regression with L2 penalty → evaluation.
`hw03_l1_model.py`  
Modified pipeline using L1 regularization (`penalty="l1"`, `solver="liblinear"`), with the same data and TF-IDF settings for a fair comparison.
`hw03_writeup.md`  
Short comparison write-up (submitted on Canvas) addressing performance, sparsity, feature correlation, label noise robustness, and historical interpretation.
Data
Both scripts expect the following JSON files produced in the Week 09 tutorial:
`train_core_vs_neg.json`
`test_core_vs_neg.json`
These files are not included in this repository. Place them in the same directory as the scripts before running.
Methods
The analysis proceeds in four main stages:
Load pre-split train/test JSON data from the Week 09 tutorial
Vectorize texts using TF-IDF (`min_df=5`, `max_df=0.95`)
Train logistic regression under both L2 and L1 penalties
Evaluate and compare: confusion matrix, classification report, ROC AUC, sparsity diagnostics, and top-15 positive/negative-weight words
Output
Each script prints evaluation metrics and the top predictive words directly to the console. The interpretive comparison is submitted separately on Canvas.
---