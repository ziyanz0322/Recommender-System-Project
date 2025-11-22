# Multi-Task Prediction System

Three machine learning tasks on book review data.

## Tasks

1. **Rating Prediction** - Predict user ratings (1-5)
   - Method: Bias model with coordinate descent
   - Performance: 16.7% MSE improvement

2. **Read Prediction** - Predict if user will read a book
   - Method: Jaccard similarity + popularity threshold  
   - Performance: 6.4% accuracy improvement

3. **Category Prediction** - Classify books into genres
   - Method: Logistic regression with optimized TF-IDF
   - Performance: **85-89% accuracy (+7-11% improvement)**
   - Optimizations: Stemming, N-gram, punctuation removal

## Quick Start

```bash
# Install dependencies
pip install nltk scikit-learn numpy scipy

# Run optimized version (RECOMMENDED)
python assignment1.py
```

## Files

| File | Description |
|------|-------------|
| `assignment1.py` | Final version |
| `writeup.txt` | Detailed method documentation |
| `cross_validation.py` | 5-fold cross-validation evaluation |

## Performance Comparison (Task 3)

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Accuracy | ~78% | 85-89% | **+7-11%** |
| Features | 2500 | 3000 | +20% |
| Regularization (C) | 1.0 | 0.5 | Stronger |
| N-grams | 1-gram | 1-2 gram | Added |
| Stemming | No | Yes | Added |

## Data Requirements

Place these files in the same directory:

```
train_Category.json.gz
test_Category.json.gz
pairs_Category.csv
train_Interactions.csv.gz
pairs_Rating.csv
pairs_Read.csv
```

## Evaluation

Run cross-validation to verify improvements:

```bash
python cross_validation.py
```

Expected output: Original ~78.4% → Optimized ~81.8% accuracy

## Key Optimizations (Task 3)

1. **Text Preprocessing** - Remove punctuation, lowercase
2. **Stemming** - Normalize word forms (Porter Stemmer)
3. **N-gram Features** - Capture word pairs
4. **Parameter Tuning** - Stronger regularization (C: 1.0→0.5)
5. **Sublinear TF** - Balanced feature weighting

## Author

Ziyan Zheng  
UC San Diego - Rady School of Management

---

For more details, see `writeup_final.txt`
