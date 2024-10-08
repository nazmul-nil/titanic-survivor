# Titanic Survival Prediction Project

This project aims to predict the survival of passengers aboard the Titanic using two machine learning algorithms: **Logistic Regression** and **Random Forest Classifier**. The project is divided into two Jupyter notebooks: one for data cleaning and Logistic Regression, and the other for Random Forest Classification. The cleaned and encoded dataset is saved in the process for reuse across models.

## Dataset Description

The Titanic dataset includes the following key features:

- **Pclass**: Passenger class (1 = First, 2 = Second, 3 = Third)
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger (some values are missing and were imputed)
- **SibSp**: Number of siblings or spouses aboard
- **Parch**: Number of parents or children aboard
- **Ticket**: Ticket reference number
- **Fare**: The ticket fare
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- **Survived**: The target variable indicating if the passenger survived (0 = No, 1 = Yes)

## Overview

The goal of this project is to predict which passengers survived the Titanic disaster using machine learning. The dataset used is the famous **Titanic dataset** from [Kaggle](https://www.kaggle.com/c/titanic).

The project follows a typical machine learning workflow:
1. **Data Cleaning and Preprocessing**: Handling missing values, encoding categorical features, and feature engineering.
2. **Model Building**: Training and tuning models to predict passenger survival.
3. **Evaluation**: Assessing model performance through accuracy, precision, recall, and other metrics.

Two models are built:
- **Logistic Regression**: A linear model used for binary classification.
- **Random Forest Classifier**: An ensemble learning method using decision trees.

## Notebooks

### 1. Titanic Survival Logistic Regression Model
- **File**: `titanic-survival-logistic-regression-model.ipynb`
- **Description**: 
  - This notebook is responsible for data cleaning, handling missing values, encoding categorical data, feature engineering, and training a **Logistic Regression** model.
  - The cleaned and encoded dataset is saved in the `data` folder for further use.
  - The notebook performs hyperparameter tuning for the Logistic Regression model using `GridSearchCV`, and the best model is selected based on performance.
  
#### Steps:
1. **Data Cleaning**: 
   - Missing values handled (Age, Embarked, etc.)
   - Categorical variables are encoded using One Hot Encoding and Label Encoding.
2. **Feature Engineering**: 
   - New features such as `Title`, `FamilySize`, and `AgeBin` are created to improve the model.
3. **Modeling**: 
   - A **Logistic Regression** model is built and evaluated on accuracy, precision, recall, and F1-score.
4. **Hyperparameter Tuning**: 
   - Hyperparameters are optimized using `GridSearchCV`, and the best model is saved.
  
- **Output**: 
   - The cleaned and encoded dataset is saved as `titanic_cleaned_encoded.csv` in the `data` folder.
   - The results include model performance metrics and a tuned Logistic Regression model.

### 2. Titanic Survival Random Forest Model
- **File**: `titanic-survival-random-forest-model.ipynb`
- **Description**:
  - This notebook loads the cleaned and encoded dataset from the `data` folder and trains a **Random Forest Classifier**.
  - Model performance is evaluated using accuracy, precision, recall, and confusion matrix.
  - Optionally, hyperparameter tuning can be applied here to improve the Random Forest model.

#### Steps:
1. **Load Cleaned Data**: 
   - The cleaned and encoded dataset (`titanic_cleaned_encoded.csv`) is loaded for further analysis.
2. **Modeling**: 
   - A **Random Forest Classifier** is trained on the data.
   - The model is evaluated using standard classification metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

- **Output**: 
   - The Random Forest model is built and evaluated, providing insights into feature importance and classification metrics.

## References

- The dataset is sourced from the [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data).
