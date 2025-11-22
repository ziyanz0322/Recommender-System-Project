#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-Validation Script - Compare Original vs Optimized Versions
Evaluates Task 3 (Category Prediction) with 5-fold cross-validation
"""

import gzip
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
import re
import string
from nltk.stem import PorterStemmer
import nltk
import warnings

warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)

print("\n" + "="*100)
print(" "*25 + "📊 Cross-Validation: Original vs Optimized")
print("="*100 + "\n")

# ============================================================================
# Data Loading Functions
# ============================================================================

def readGz(path):
    """Read gzip file"""
    for l in gzip.open(path, "rt"):
        yield eval(l)


print("Step 1: Loading data...")

# Load data
dataTrain = []
count = 0
for d in readGz("train_Category.json.gz"):
    dataTrain.append(d)
    count += 1
    if count % 5000 == 0:
        print(f"   Loaded {count} samples...")

print(f"✅ Loading complete: {len(dataTrain)} samples\n")

# Extract labels and texts
y = np.array([d["genreID"] for d in dataTrain])
texts = [d.get("review_text", "") for d in dataTrain]

print(f"Data Statistics:")
print(f"   Total samples: {len(dataTrain)}")
print(f"   Number of classes: {len(set(y))}")
print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}\n")

# ============================================================================
# VERSION 1: Original Version
# ============================================================================

print("="*100)
print("🔴 Version 1: Original (Baseline)")
print("="*100)
print("Configuration:")
print("  - TF-IDF max_features: 2500")
print("  - TF-IDF max_df: 0.8")
print("  - Preprocessing: lowercase only")
print("  - N-gram: (1,1) unigrams only")
print("  - Stemming: ❌ No")
print("  - Model C: 1.0")
print("  - Model max_iter: 2000\n")

print("Running 5-fold cross-validation...")

# Original vectorizer
vectorizer_original = TfidfVectorizer(
    max_features=2500,
    min_df=2,
    max_df=0.8,
    lowercase=True,
    stop_words="english"
)

# Original model
model_original = linear_model.LogisticRegression(
    C=1.0,
    max_iter=2000,
    multi_class="multinomial",
    solver="lbfgs",
    random_state=42
)

# Transform texts
X_original = vectorizer_original.fit_transform(texts)
print(f"   TF-IDF features: {X_original.shape[1]}")

# 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores_original = cross_val_score(
    model_original, X_original, y,
    cv=kfold,
    scoring='accuracy',
    n_jobs=-1
)

print(f"\n✅ Original Version Cross-Validation Results:")
print(f"   Fold 1: {scores_original[0]:.4f}")
print(f"   Fold 2: {scores_original[1]:.4f}")
print(f"   Fold 3: {scores_original[2]:.4f}")
print(f"   Fold 4: {scores_original[3]:.4f}")
print(f"   Fold 5: {scores_original[4]:.4f}")
print(f"   {'-'*30}")
print(f"   Mean Accuracy: {scores_original.mean():.4f}")
print(f"   Std Dev: {scores_original.std():.4f}")
print(f"   95% CI: [{scores_original.mean() - 1.96*scores_original.std():.4f}, {scores_original.mean() + 1.96*scores_original.std():.4f}]")

original_mean = scores_original.mean()

# ============================================================================
# VERSION 2: Optimized Version
# ============================================================================

print("\n" + "="*100)
print("🟢 Version 2: Optimized")
print("="*100)
print("Configuration:")
print("  - TF-IDF max_features: 3000")
print("  - TF-IDF max_df: 0.85")
print("  - Preprocessing: remove punctuation + lowercase")
print("  - N-gram: (1,2) unigrams + bigrams")
print("  - Stemming: ✅ Porter Stemming")
print("  - sublinear_tf: ✅ True")
print("  - Model C: 0.5")
print("  - Model max_iter: 3000\n")

# Initialize Stemmer
stemmer = PorterStemmer()

def preprocess_text(text):
    """Preprocess text"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[{}]+'.format(re.escape(string.punctuation)), ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def custom_tokenizer(text):
    """Custom tokenizer with stemming"""
    text = preprocess_text(text)
    words = text.split()
    return [stemmer.stem(word) for word in words]

print("Preprocessing texts...")
texts_preprocessed = [preprocess_text(t) for t in texts]
print(f"✅ Preprocessing complete")

print("Running 5-fold cross-validation...")

# Optimized vectorizer
vectorizer_optimized = TfidfVectorizer(
    max_features=3000,
    min_df=2,
    max_df=0.85,
    tokenizer=custom_tokenizer,
    lowercase=False,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

# Optimized model
model_optimized = linear_model.LogisticRegression(
    C=0.5,
    max_iter=3000,
    multi_class="multinomial",
    solver="lbfgs",
    random_state=42
)

# Transform texts
X_optimized = vectorizer_optimized.fit_transform(texts_preprocessed)
print(f"   TF-IDF features: {X_optimized.shape[1]}")

# 5-fold cross-validation
scores_optimized = cross_val_score(
    model_optimized, X_optimized, y,
    cv=kfold,
    scoring='accuracy',
    n_jobs=-1
)

print(f"\n✅ Optimized Version Cross-Validation Results:")
print(f"   Fold 1: {scores_optimized[0]:.4f}")
print(f"   Fold 2: {scores_optimized[1]:.4f}")
print(f"   Fold 3: {scores_optimized[2]:.4f}")
print(f"   Fold 4: {scores_optimized[3]:.4f}")
print(f"   Fold 5: {scores_optimized[4]:.4f}")
print(f"   {'-'*30}")
print(f"   Mean Accuracy: {scores_optimized.mean():.4f}")
print(f"   Std Dev: {scores_optimized.std():.4f}")
print(f"   95% CI: [{scores_optimized.mean() - 1.96*scores_optimized.std():.4f}, {scores_optimized.mean() + 1.96*scores_optimized.std():.4f}]")

optimized_mean = scores_optimized.mean()

# ============================================================================
# Comparison Analysis
# ============================================================================

print("\n" + "="*100)
print("📈 Comparison Analysis - Original vs Optimized")
print("="*100 + "\n")

improvement_absolute = optimized_mean - original_mean
improvement_relative = (improvement_absolute / original_mean) * 100

print(f"{'Metric':<35} {'Original':<20} {'Optimized':<20} {'Improvement':<20}")
print("-" * 95)
print(f"{'Mean Accuracy':<35} {original_mean:<20.4f} {optimized_mean:<20.4f} {improvement_absolute:>+.4f} ({improvement_relative:+.2f}%)")
print(f"{'Std Dev':<35} {scores_original.std():<20.4f} {scores_optimized.std():<20.4f} {scores_optimized.std()-scores_original.std():>+.4f}")
print(f"{'Max Accuracy':<35} {scores_original.max():<20.4f} {scores_optimized.max():<20.4f} {scores_optimized.max()-scores_original.max():>+.4f}")
print(f"{'Min Accuracy':<35} {scores_original.min():<20.4f} {scores_optimized.min():<20.4f} {scores_optimized.min()-scores_original.min():>+.4f}")
print(f"{'TF-IDF Features':<35} {X_original.shape[1]:<20} {X_optimized.shape[1]:<20}")

# ============================================================================
# Statistical Significance Test
# ============================================================================

print("\n" + "-" * 100)
print("\n📊 Statistical Significance Test (Paired t-test):\n")

from scipy import stats

# Paired t-test
t_stat, p_value = stats.ttest_rel(scores_optimized, scores_original)

print(f"   t-statistic: {t_stat:.4f}")
print(f"   p-value: {p_value:.4f}")

if p_value < 0.05:
    print(f"   ✅ Result is significant (p < 0.05)")
    print(f"   💡 The accuracy improvement is statistically significant!")
elif p_value < 0.10:
    print(f"   ⚠️ Result is marginally significant (p < 0.10)")
else:
    print(f"   ❌ Result is not significant (p >= 0.10)")
    print(f"   💡 May need more data to verify improvement")

# ============================================================================
# Optimization Details
# ============================================================================

print("\n" + "="*100)
print("🔍 Optimization Details")
print("="*100 + "\n")

details = [
    ("Text Preprocessing",
     "lowercase only",
     "remove punctuation + lowercase + clean whitespace"),

    ("Word Normalization",
     "none (raw word forms)",
     "Porter Stemming (consolidate word forms)"),

    ("Feature Type",
     "1-gram (unigrams)",
     "1-gram + 2-gram (unigrams + bigrams)"),

    ("TF-IDF Parameters",
     "max_features=2500, max_df=0.8",
     "max_features=3000, max_df=0.85, sublinear_tf=True"),

    ("Model Regularization",
     "C=1.0 (weak regularization)",
     "C=0.5 (strong regularization)"),

    ("Iterations",
     "max_iter=2000",
     "max_iter=3000"),
]

for i, (aspect, original, optimized) in enumerate(details, 1):
    print(f"{i}. {aspect}")
    print(f"   Original: {original}")
    print(f"   Optimized: {optimized}")
    print()

# ============================================================================
# Results Summary
# ============================================================================

print("="*100)
print("✨ Results Summary")
print("="*100 + "\n")

print(f"🎯 Accuracy Improvement:")
print(f"   Absolute: {improvement_absolute:+.4f} ({improvement_absolute*100:+.2f}%)")
print(f"   Relative: {improvement_relative:+.2f}%")

if improvement_relative >= 8:
    rating = "🌟🌟🌟 Excellent!"
elif improvement_relative >= 5:
    rating = "🌟🌟 Very Good"
elif improvement_relative >= 2:
    rating = "🌟 Good"
else:
    rating = "⚠️ Limited"

print(f"   Rating: {rating}\n")

print(f"📌 Key Findings:")

if scores_optimized.std() < scores_original.std():
    print(f"   ✅ Optimized version is more stable (smaller std dev)")
else:
    print(f"   ℹ️ Slightly higher variance (normal trade-off)")

if scores_optimized.min() > scores_original.min():
    print(f"   ✅ Optimized version has higher minimum accuracy")

if p_value < 0.05:
    print(f"   ✅ Improvement is statistically significant (p={p_value:.4f})")

# ============================================================================
# Recommendations
# ============================================================================

print("\n" + "="*100)
print("💡 Recommendations")
print("="*100 + "\n")

if improvement_relative < 2:
    print("   1. Current improvement is limited. Consider:")
    print("      - Increasing training data")
    print("      - Trying different models (SVM, Random Forest)")
    print("      - Advanced feature engineering")
    print("      - Hyperparameter tuning with GridSearchCV")
elif improvement_relative < 5:
    print("   1. Improvement is decent. Consider:")
    print("      - Ensemble methods")
    print("      - Further parameter tuning")
    print("      - Additional features")
else:
    print("   1. ✅ Optimization is successful!")
    print("      - Recommend using optimized version")
    print("      - assignment1_optimized.py as final submission")

print("\n   2. For further improvement:")
print("      - Use GridSearchCV for optimal parameters")
print("      - Try Lemmatization instead of Stemming")
print("      - Add more N-grams (3-grams)")
print("      - Combine multiple text features (Word2Vec, FastText)")

# ============================================================================
# Conclusion
# ============================================================================

print("\n" + "="*100)
print("🎓 Conclusion")
print("="*100 + "\n")

print(f"✅ Optimized version performance on 5-fold cross-validation:")
print(f"   Accuracy: {optimized_mean:.4f} (Original: {original_mean:.4f})")
print(f"   Improvement: {improvement_relative:+.2f}%")

if improvement_relative >= 5:
    print(f"\n🎉 Optimization successful! Use optimized version (assignment1_optimized.py)")
else:
    print(f"\n⚠️ Improvement is limited. Consider further optimization.")

print("\n" + "="*100 + "\n")
