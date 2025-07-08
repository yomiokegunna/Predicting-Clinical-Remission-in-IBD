# Predicting-Clinical-Remission-in-Crohn's disease

This repository contains the code and methodology for predicting clinical remission in Crohn’s disease using two distinct modeling approaches: Bayesian Networks and Machine Learning Models . The project is structured to separate the implementation of Bayesian Networks (both Expert-defined and Data-driven) from traditional machine learning models (Naïve Bayes, Logistic Regression, and XGBoost). This separation ensures clarity and modularity, making it easier to navigate and extend the codebase.

Project Structure
The repository is organized into two main directories:

bayesian_networks/ : Contains all code related to the development, evaluation, and optimization of Bayesian Network models.
machine_learning_models/ : Contains all code related to the implementation and evaluation of traditional machine learning models.
1. Bayesian Networks (bayesian_networks/)
This directory focuses on the construction, refinement, and evaluation of Bayesian Network models, including both Expert-defined and Data-driven approaches.

Key Files and Scripts
expert_bn.py :
Implements the Expert-defined Bayesian Network using domain knowledge.
Includes functions to define edges, visualize the network, and evaluate performance metrics.
Outputs a JSON file (expert_model.json) for visualization in tools like Cytoscape.
data_driven_bn.py :
Implements the Data-driven Bayesian Network using structure learning algorithms (e.g., Hill-Climb Search with BIC, K2, BDeu, and BDs scoring methods).
Includes node removal analysis to identify features that negatively impact model performance.
Visualizes the refined Bayesian Network after removing detrimental nodes.
utils_bn.py :
Utility functions for preprocessing data, fitting Bayesian Networks, and evaluating predictions (e.g., ROC-AUC, confusion matrices).
results_bn.md :
Summary of key findings and performance metrics for both Expert and Data-driven Bayesian Networks.

2. Machine Learning Models (machine_learning_models/)
This directory focuses on the implementation and evaluation of traditional machine learning models, including Naïve Bayes, Logistic Regression, and XGBoost.

Key Files and Scripts
naive_bayes.py :
Implements the Naïve Bayes model using categorical feature encoding.
Includes performance evaluation and probability predictions.
logistic_regression.py :
Implements Logistic Regression with stepwise feature selection to identify the most important predictors.
Includes scaling of features and hyperparameter tuning.
xgboost_model.py :
Implements XGBoost with hyperparameter optimization using Randomized Search.
Evaluates performance metrics and generates feature importance plots.
ensemble_voting_classifier.py :
Combines predictions from all five models (Expert BN, Data-driven BN, Naïve Bayes, Logistic Regression, XGBoost) using weighted voting schemes.
Evaluates ensemble performance across different weighting strategies (accuracy-based, AUC-based, sensitivity-based, specificity-based, balanced weights).
utils_ml.py :
Utility functions for preprocessing data, training models, and evaluating predictions (e.g., ROC-AUC, confusion matrices, classification reports).
results_ml.md :
Summary of key findings and performance metrics for all machine learning models and the ensemble approach.

Data Preprocessing
Both Bayesian Networks and machine learning models rely on preprocessed data. The preprocessing pipeline includes:

Comorbidity Index Calculation : Using predefined weights to compute the Charlson Comorbidity Index (CCI).
Medication Usage Encoding : Categorizing medication usage as binary indicators or NaN for missing values.
Handling Missing Values : Mode imputation for categorical features and SMOTENC for addressing class imbalance.
Feature Engineering : Deriving features such as Harvey-Bradshaw Index (HBI), age groups, and EIM scores.
The preprocessing scripts are shared across both directories and can be found in the preprocessing section.

Results and Metrics
Performance metrics for all models are reported in their respective directories:

Bayesian Networks : Accuracy, AUC-ROC, sensitivity, specificity, and confusion matrices.
Machine Learning Models : Accuracy, AUC-ROC, precision, recall, F1-score, and confusion matrices.
Weighted ensemble voting results are also included, highlighting the best-performing weighting scheme for each metric.

Dependencies
Python 3.x
Libraries: pandas, numpy, scikit-learn, xgboost, pgmpy, networkx, matplotlib, imbalanced-learn
