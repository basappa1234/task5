# Heart Disease Prediction - Decision Tree & Random Forest

## Overview
This project predicts heart disease presence using machine learning classification models.  
It uses a dataset (`heart.csv`) containing patient health metrics, and compares two algorithms:
- **Decision Tree Classifier** (with depth tuning to avoid overfitting)
- **Random Forest Classifier** (ensemble learning)

## Steps Implemented
1. **Import Libraries**  
   pandas, numpy, matplotlib for data handling & plotting;  
   scikit-learn for model training, evaluation, and visualization.

2. **Load & Prepare Data**  
   - Reads `heart.csv`.
   - Splits into features (`X`) and target (`y`).
   - Splits dataset into training (70%) and testing (30%) sets.

3. **Decision Tree Model**
   - Trains an unpruned Decision Tree.
   - Visualizes the tree structure.
   - Evaluates accuracy on training and test sets.
   - Tests multiple `max_depth` values to analyze overfitting/underfitting.
   - Finds the optimal tree depth.

4. **Random Forest Model**
   - Trains with 100 decision trees (`n_estimators=100`).
   - Compares accuracy with Decision Tree.
   - Evaluates performance on the test set.

5. **Results**
   - Plots accuracy vs. tree depth for Decision Tree.
   - Prints best `max_depth` and corresponding accuracy.
   - Shows Random Forest accuracy for comparison.

## Key Concepts Covered
- Train/test split for unbiased evaluation.
- Decision Tree basics and overfitting prevention using `max_depth`.
- Random Forest as an ensemble method to improve accuracy and stability.
- Model performance evaluation using accuracy scores.
- Data visualization for better interpretability.

