# Credit Card Fraud Detection

## Introduction

Credit card fraud is a significant issue affecting financial institutions and consumers worldwide. Timely detection of fraudulent transactions is essential to minimize financial losses and maintain trust with customers. This project presents a comprehensive analysis of credit card transaction data to develop effective fraud detection models using machine learning techniques.

## Data Preparation

### Data Description
- Dataset Size: 38,587 transactions × 31 features
- Features: Includes transaction amount ('Amount') and class labels ('Class') indicating fraudulent or non-fraudulent transactions.

### Handling Missing Values
- Missing values are detected in several columns and appropriately addressed to ensure data integrity and model robustness.

### Feature Engineering
- Interaction features are created by multiplying two existing features ('V1' and 'V2') to enhance the dataset with additional information.

## Exploratory Data Analysis (EDA)

### Feature Distribution
- Histograms visualize the distribution of transaction amounts and explore variability across features.

### Correlation Analysis
- Correlation between features ('V1' to 'V28') and the target variable ('Class') is examined to identify potential relationships and predictive power.

### Class Imbalance
- The imbalance between fraudulent and non-fraudulent transactions is analyzed to understand the dataset's class distribution and potential challenges during model training.

## Model Development

### Data Splitting
- The dataset is divided into training and testing sets to facilitate model training and evaluation.

### Model Training
- Multiple machine learning models, including Logistic Regression, Random Forest, and Support Vector Machine, are trained on the training data to learn patterns and relationships.

### Hyperparameter Tuning
- GridSearchCV is employed to optimize hyperparameters of the Random Forest classifier, enhancing model performance.

## Model Evaluation

### Evaluation Metrics
- Model performance is assessed using various classification metrics, including precision, recall, F1-score, and ROC-AUC score, on the test set.

### Model Comparison
- Performance of different models is compared based on evaluation metrics to identify the most effective fraud detection model.

## Results and Analysis

1. **Data Preparation:**
   - **Numerical Stats:**
     - Total number of transactions: 38,587
     - Number of features: 31
     - Class distribution (before SMOTE):
       - Non-fraudulent transactions (Class 0): 38,483
       - Fraudulent transactions (Class 1): 103
     - Class distribution (after SMOTE):
       - Balanced distribution: 38,483 for both Class 0 and Class 1

   - **Visualizations:**
     - Histograms: Distribution of transaction amounts for fraudulent and non-fraudulent transactions.
     - Bar Chart: Correlation between features (V1-V28) and Class.
     - Box Plot: Distribution of transaction amounts for fraudulent and non-fraudulent transactions.
     - Violin Plot: Distribution of each feature (V1-V28) for fraudulent and non-fraudulent transactions.
     - Box Plot: Distribution of each feature (V1-V28) to identify outliers.

2. **Model Evaluation:**
   - **Numerical Stats:**
     - Logistic Regression:
       - Accuracy: 99.9%
       - ROC-AUC Score: 99.4%
     - Random Forest:
       - Accuracy: 100%
       - ROC-AUC Score: 99.99%
     - Support Vector Machine:
       - Accuracy: 100%
       - ROC-AUC Score: 99.97%
     - Cross-validation mean accuracy: 99.99%

   - **Visualizations:**
     - Classification Report: Precision, recall, F1-score, and support for each class.
     - ROC Curve: Graphical representation of true positive rate (recall) vs. false positive rate for each model.
     - SHAP Summary Plot: Visualization of feature importance using SHAP values to interpret the Random Forest model.

3. **Additional Insights:**
   - Feature Engineering: Interaction feature (V1_V2_interaction) improves Random Forest model performance.
   - Cross-validation demonstrates model robustness with consistently high accuracy scores across folds.
   - Class imbalance is effectively addressed through SMOTE, resulting in a balanced distribution of fraudulent and non-fraudulent transactions.

4. **Overall Analysis:**
   - All models achieve exceptional performance with high accuracy and ROC-AUC scores, indicating effective fraud detection capabilities.
   - Random Forest model outperforms others slightly in terms of ROC-AUC score, suggesting better discriminatory power in ranking and differentiating between transactions.
   - EDA findings and model interpretability using SHAP values provide insights into feature importance and model decision-making processes.
   - The balanced class distribution after SMOTE enhances model training and evaluation, contributing to improved fraud detection accuracy.
