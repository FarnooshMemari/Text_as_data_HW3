import json, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

os.makedirs("output/l2", exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────
with open("data/train_core_vs_neg.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open("data/test_core_vs_neg.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

train_texts = [doc[0] for doc in train_data]
train_labels = [doc[1] for doc in train_data]
test_texts = [doc[0] for doc in test_data]
test_labels = [doc[1] for doc in test_data]

# ── TF-IDF vectorization (same settings as Week 09) ─────────────────
vectorizer = TfidfVectorizer(min_df=5, max_df=0.95)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)
y_train = np.array(train_labels)
y_test = np.array(test_labels)

print(f"Vocabulary size: {X_train.shape[1]}")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ── Train logistic regression with L2 ───────────────────────────────
clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# ── Evaluation ───────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
report = classification_report(
    y_test, y_pred, target_names=["NEG (0)", "CORE (1)"], digits=4
)
auc = roc_auc_score(y_test, y_prob)
n_nonzero = np.sum(clf.coef_[0] != 0)
n_total = clf.coef_.shape[1]

print(f"\n=== Confusion Matrix ===\n{cm}")
print(f"\n=== Classification Report ===\n{report}")
print(f"=== ROC AUC: {auc:.4f} ===")
print(f"=== Sparsity: {n_nonzero} / {n_total} coefficients are non-zero ===")

# ── Top words ────────────────────────────────────────────────────────
feature_names = vectorizer.get_feature_names_out()
coefs = clf.coef_[0]
sorted_idx = np.argsort(coefs)

top_pos_idx = sorted_idx[-15:][::-1]
top_neg_idx = sorted_idx[:15]

print("\n=== Top 15 Positive-Weight Words (most predictive of CORE = 1) ===")
for i in top_pos_idx:
    print(f"  {feature_names[i]:25s}  {coefs[i]:+.4f}")

print("\n=== Top 15 Negative-Weight Words (most predictive of NEG = 0) ===")
for i in top_neg_idx:
    print(f"  {feature_names[i]:25s}  {coefs[i]:+.4f}")

# ── Bar chart: top 15 positive words ────────────────────────────────
pos_words = [feature_names[i] for i in top_pos_idx]
pos_vals = [coefs[i] for i in top_pos_idx]

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(pos_words[::-1], pos_vals[::-1], color="#2a9d8f")
ax.set_xlabel("Coefficient Weight")
ax.set_title("L2 — Top 15 Positive-Weight Words (CORE = 1)")
plt.tight_layout()
plt.savefig("output/l2/top15_positive_words.png", dpi=150)
plt.close()

# ── Bar chart: top 15 negative words ────────────────────────────────
neg_words = [feature_names[i] for i in top_neg_idx]
neg_vals = [coefs[i] for i in top_neg_idx]

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(neg_words[::-1], neg_vals[::-1], color="#e76f51")
ax.set_xlabel("Coefficient Weight")
ax.set_title("L2 — Top 15 Negative-Weight Words (NEG = 0)")
plt.tight_layout()
plt.savefig("output/l2/top15_negative_words.png", dpi=150)
plt.close()

# ── Save metrics to text file ───────────────────────────────────────
with open("output/l2/metrics.txt", "w") as f:
    f.write("L2 Baseline — Evaluation Results\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Confusion Matrix:\n{cm}\n\n")
    f.write(f"Classification Report:\n{report}\n")
    f.write(f"ROC AUC: {auc:.4f}\n\n")
    f.write(f"Non-zero coefficients: {n_nonzero}\n")
    f.write(f"Total coefficients: {n_total}\n")
    f.write(f"Sparsity ratio: {n_nonzero / n_total:.4f}\n")

print("\nAll outputs saved to output/l2/")
