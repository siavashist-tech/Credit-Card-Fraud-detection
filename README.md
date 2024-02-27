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

### Data Preparation
- Numerical Stats: Total transactions, feature count, and class distribution before and after SMOTE.
- Visualizations: Histograms, bar chart, box plot, and violin plot to explore feature distributions.

### Model Evaluation
- Numerical Stats: Performance metrics (accuracy, ROC-AUC score) for Logistic Regression, Random Forest, and Support Vector Machine.
- Visualizations: Classification report, ROC curve, and SHAP summary plot to interpret model performance.

### Additional Insights
- Feature Engineering: Interaction feature improves Random Forest model performance.
- Cross-validation demonstrates model robustness with consistently high accuracy scores.
- SMOTE effectively addresses class imbalance, enhancing model training and evaluation.

### Overall Analysis
- All models achieve exceptional performance with high accuracy and ROC-AUC scores.
- Random Forest model slightly outperforms others, suggesting better discriminatory power.
- EDA findings and SHAP values provide insights into feature importance and model decision-making processes.
- Balanced class distribution after SMOTE contributes to improved fraud detection accuracy.

