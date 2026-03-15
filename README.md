# Ames Housing Price Prediction: Advanced Feature Engineering

This project demonstrates a comprehensive feature engineering pipeline for the Ames Housing dataset, originally completed as part of the Kaggle Feature Engineering course.

## Project Overview
The goal of this project is to move beyond basic data cleaning and apply advanced unsupervised and supervised techniques to uncover hidden patterns in the data, ultimately improving the predictive power of a Gradient Boosting model (XGBoost).

## Techniques Applied

- **Mutual Information (MI):** Used MI scores to rank 70+ features and identify the most impactful predictors like `OverallQual` and `GrLivArea`.
- **Mathematical Transforms:** Created synthetic features such as `LivLotRatio` and `Spaciousness` to capture non-linear relationships.
- **Unsupervised Learning (K-Means Clustering):** Grouped houses based on physical dimensions (Area, Basement size) to create a new `Cluster` feature.
- **Dimensionality Reduction (PCA):** Applied Principal Component Analysis to highly correlated features (Garage, Year Built) to capture the "Size" and "Age" variance in fewer components.
- **Target Encoding:** Encoded high-cardinality categorical features like `Neighborhood` using smoothed target statistics to help the model handle location-based trends.

## Results
By applying these feature engineering steps, the model's performance was significantly enhanced.
- **Baseline RMSLE:** 0.14321
- **Final Optimized RMSLE:** 0.11985
- 
## Repository Structure
- `feature_engineering.py`: The main Python script containing the full pipeline.
- `requirements.txt`: List of necessary libraries (pandas, scikit-learn, xgboost, etc.).

