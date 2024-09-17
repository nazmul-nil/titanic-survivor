# Titanic Survivor Prediction

This project focuses on predicting whether a passenger on the Titanic would survive based on a set of features such as age, gender, passenger class, and others. Using the Titanic dataset, we employ Logistic Regression as the classification model to predict the survival status.

## Overview

The goal of this project is to create a machine learning model that predicts the survival of Titanic passengers using Logistic Regression. The dataset contains various features, and we preprocess these before feeding them into the model. Finally, we evaluate the model using the accuracy score to measure its performance.

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

## Model

We used `LogisticRegression`, a linear model used for binary classification tasks, to predict the survival outcome of passengers.

Key steps include:
- Data preprocessing (handling missing values, encoding categorical features)
- Feature selection
- Training the Logistic Regression model

## Evaluation

The model's performance was evaluated using the `accuracy_score`, which represents the ratio of correct predictions to the total number of predictions made.

## References

- The dataset is sourced from the [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data).
- Logistic Regression implementation from [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
