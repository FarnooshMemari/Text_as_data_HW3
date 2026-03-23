import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Load saved train/test JSON files from Week 09
with open("data/train_core_vs_neg.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open("data/test_core_vs_neg.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

train_texts = [doc[0] for doc in train_data]
train_labels = [doc[1] for doc in train_data]
test_texts = [doc[0] for doc in test_data]
test_labels = [doc[1] for doc in test_data]

# TF-IDF vectorization (same settings as Week 09)
vectorizer = TfidfVectorizer(max_df=0.95, min_df=5)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)
y_train = np.array(train_labels)
y_test = np.array(test_labels)

print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")

# L2 logistic regression (scikit-learn default)
clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print()

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["NEG (0)", "CORE (1)"]))

auc = roc_auc_score(y_test, y_prob)
print(f"=== ROC AUC: {auc:.4f} ===\n")

coefs = clf.coef_[0]
n_nonzero = np.sum(coefs != 0)
print(f"=== Sparsity: {n_nonzero} / {len(coefs)} coefficients are non-zero ===\n")

feature_names = vectorizer.get_feature_names_out()
sorted_idx = np.argsort(coefs)

print("=== Top 15 Positive-Weight Words (most predictive of CORE = 1) ===")
for i in sorted_idx[-15:][::-1]:
    print(f"  {feature_names[i]:25s}  {coefs[i]:.4f}")

print("\n=== Top 15 Negative-Weight Words (most predictive of NEG = 0) ===")
for i in sorted_idx[:15]:
    print(f"  {feature_names[i]:25s}  {coefs[i]:.4f}")
