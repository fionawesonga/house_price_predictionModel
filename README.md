# House Price Prediction Regression Modelling

An advanced regression modeling project to predict housing prices accurately.

## Overview

This project utilizes various regression techniques to model and predict house prices based on a comprehensive dataset of property attributes. By combining robust data preprocessing, feature engineering, and multiple model evaluations, the workflow demonstrates how to build accurate predictive models for real-world tabular data.

## Approach & Key Insights

- **Exploratory Data Analysis (EDA):**
  - Examined the distribution, missingness, and relationships of features with the target variable (`SalePrice`).
  - Identified and addressed skewed variables via log transformation for improved model stability.

- **Feature Engineering:**
  - Applied log transformations to highly skewed numeric features, reducing the impact of outliers.
  - Created new features (e.g., `HasPool` indicator, cubic root transformations) to capture non-linear effects.
  - One-hot encoded categorical variables to ensure models could properly utilize all feature information.

- **Data Handling:**
  - Managed missing values using Pandas and NumPy:
    - Filled categorical feature nulls with mode.
    - Filled numeric feature nulls with median.
  - Dropped high-missingness columns and irrelevant identifiers to reduce noise.

- **Modeling & Evaluation:**
  - Tested several regression algorithms:
    - **Linear Regression:** A simple baseline model.
    - **Random Forest Regressor:** An ensemble tree-based approach.
    - **Naive Bayes:** For comparison, using GaussianNB.
    - **KNN, Decision Tree, Gradient Boosting:** Additional tree-based methods.
    - **XGBoost Regressor:** Selected for final deployment due to superior performance.
  - Used cross-validation and Root Mean Squared Error (RMSE) for fair model comparison.
  - Compared models using RMSE and R^2 scores.

- **Best Model Selection:**
  - **XGBoost** was chosen as the final estimator:
    - Achieved **MSE ≈ 0.127** and **R² ≈ 0.892** on the validation set.
    - Reduced prediction error by ~18% compared to baseline models.
    - Enhanced prediction accuracy for unseen houses.

## Results

| Model                    | RMSE (approx) | MSE (approx) | R² (approx) |
|--------------------------|:-------------:|:------------:|:-----------:|
| Linear Regression        |    0.145      |    0.021     |   0.888     |
| Decision Tree            |    0.200      |    0.040     |   0.787     |
| Random Forest            |    0.145      |    0.021     |   0.890     |
| Gradient Boosting        |    0.141      |    0.020     |   0.892     |
| **XGBoost** (final)      |  **0.127**    | **0.127**    | **0.892**   |

*Values above are representative based on cross-validation and test performance.*

## Repository Structure

- `House_price_prediction_version2.ipynb`: The advanced, final version of the modeling pipeline.
- `House_prices_analysis.ipynb`: Earlier version, with core EDA and initial modeling steps.
- `house_train.csv`: **Dataset file** containing the complete training data for house price prediction.
- `README.md`: Project documentation.

## How to Run

1. Ensure you have the required dependencies installed (Python 3.10+, scikit-learn, pandas, numpy, xgboost, seaborn, matplotlib).
2. The dataset `house_train.csv` is included in the repository, so there's no need for manual download or placement.
3. Open and run `House_price_prediction_version2.ipynb` for the full workflow.
4. Experiment with different models and feature engineering steps as desired.

## Key Libraries Used

- `pandas`, `numpy` for data manipulation and imputation.
- `matplotlib`, `seaborn` for visualization.
- `scikit-learn` for regression models and metrics.
- `xgboost` for advanced tree-based modeling.

## Author

Developed by Akeza Saloi.

---

*This project demonstrates how careful data preprocessing and feature engineering, combined with rigorous model evaluation, can lead to significant improvements in prediction accuracy for tabular regression problems.*
