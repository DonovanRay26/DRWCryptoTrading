# DRW Crypto Market Competition - Ensemble Approach

## Competition Results
This project placed in the top 5.4% of the DRW Crypto Market competition using an ensemble of machine learning models for cryptocurrency price prediction.

## Overview
An ensemble model that combines multiple machine learning approaches to predict cryptocurrency market movements. The approach uses different model types to capture various patterns in the data while preventing overfitting.

## Ensemble Architecture

### Models Used
1. **LightGBM (40% weight)** - Gradient boosting with conservative parameters
2. **CatBoost (30% weight)** - Advanced gradient boosting with categorical handling
3. **Ridge Regression (20% weight)** - Linear model with strong regularization
4. **TabNet (15% weight)** - Deep neural network for tabular data
5. **Extra Trees (10% weight)** - Ensemble of extremely randomized trees

### Design Approach
- Conservative model configurations to prevent overfitting
- Diverse model types to capture different market patterns
- Dynamic weight adjustment based on validation performance
- Feature selection to reduce dimensionality by ~78%

## Feature Selection

### Selection Criteria
The feature selection process uses multiple metrics optimized for cryptocurrency data:

- **Mutual Information (30%)** - Non-linear relationships
- **Spearman Correlation (25%)** - Monotonic patterns
- **Pearson Correlation (20%)** - Linear relationships
- **F-Statistics (15%)** - Linear model relevance
- **Lasso Regularization (5%)** - Sparse selection
- **Temporal Stability (5%)** - Feature consistency over time

### Feature Reduction Process
- Started with 895 features
- Removed 275 highly correlated features (threshold: 0.95)
- Final selection: 200 features (77.7% reduction)

## Model Training

### Hyperparameters
All models use conservative settings to prevent overfitting:

- **LightGBM**: Learning rate 0.005, max leaves 10, strong L1/L2 regularization
- **CatBoost**: Learning rate 0.005, depth 3, high L2 regularization
- **Ridge**: Alpha 50.0 for maximum regularization
- **TabNet**: 8 hidden units, early stopping, momentum 0.3
- **Extra Trees**: Max depth 4, 25 estimators

### Training Strategy
- 80/20 train/validation split
- Early stopping with patience-based stopping
- Performance metrics: correlation coefficient and RMSE

## Weighting System

### Weight Calculation
The ensemble combines:
- Base weights (70% influence)
- Performance-based weights (30% influence)

Weights are adjusted based on:
- Pearson correlation with actual values
- Model stability
- Ensemble diversity

## Data Processing

### Handling Missing Data
- Replace infinite values with NaN
- Forward fill for time series data
- Fill remaining NaNs with 0

### Dataset Sizes
- Training: 525,887 samples × 200 features
- Test: 538,150 samples × 200 features

## Technical Implementation

### Dependencies
- LightGBM, CatBoost, scikit-learn
- PyTorch, PyTorch TabNet
- pandas, numpy, pyarrow
- matplotlib, seaborn

### Memory Management
- Garbage collection after large operations
- Efficient feature selection algorithms
- Subsampling for intensive computations

## Results

### Model Performance
- **LightGBM**: Correlation 0.0164, RMSE 1.0405
- **CatBoost**: Correlation 0.0055, RMSE 1.0405
- **Ridge**: Correlation 0.0891, RMSE 1.1470
- **TabNet**: Correlation 0.0305, RMSE 1.0717
- **Extra Trees**: Correlation 0.0256, RMSE 1.0692

### Predictions
- Range: [-1.7145, 2.4904]
- Mean: -0.0031
- Standard deviation: 0.1939

## Project Files

```
DRWCryptoTrading/
├── ensemble.ipynb          # Main implementation
├── train.parquet          # Training data
├── test.parquet           # Test data
└── ensemble_submission.csv # Competition submission
```

## Key Insights

### What Worked
- Conservative model configurations prevented overfitting
- Feature selection improved model stability
- Ensemble diversity captured different patterns
- Adaptive weighting optimized performance

### Challenges
- High-dimensional feature space (895 → 200 features)
- Cryptocurrency market volatility
- Memory constraints with large datasets
- Preventing overfitting in competitive environments

---

This ensemble approach balanced model complexity with generalization, achieving top performance through careful feature selection, conservative model design, and intelligent ensemble weighting.
