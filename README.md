# Titanic KNN Project

Overview

This project uses the K-Nearest Neighbors (KNN) algorithm to predict the class (Pclass) of Titanic passengers based on several features. By analyzing variables such as passenger demographics, ticket information, and more, the goal is to classify which class a passenger belonged to.

Features Used

The dataset includes the following variables:

PassengerId: Unique ID for each passenger.

Survived: Survival status (not used in prediction).

Pclass: Passenger class (target variable).

Name: Name of the passenger.

Sex: Gender of the passenger.

Age: Age of the passenger.

SibSp: Number of siblings/spouses aboard.

Parch: Number of parents/children aboard.

Ticket: Ticket number.

Fare: Ticket fare.

Cabin: Cabin number.

Embarked: Port of embarkation.

Data Preprocessing

To reduce noise and prepare the data for modeling, the following preprocessing steps were applied:

Missing Values: Imputed missing values using the mean or mode where appropriate.

Outlier Removal: Clipped numerical variables to remove extreme values using quantile thresholds.

Encoding: Converted categorical variables (e.g., Sex, Embarked) to numerical values using one-hot encoding.

Feature Selection: Dropped irrelevant or redundant features (e.g., PassengerId, Name, Ticket, Cabin).

Modeling

    Several models were developed and evaluated:

    Baseline KNN Model: Used default parameters to establish a baseline.

    Grid Search: Optimized the n_neighbors parameter using GridSearchCV.

    Randomized Search: Conducted a broader search with RandomizedSearchCV to explore additional configurations.

    Bayesian Optimization: Fine-tuned hyperparameters using BayesSearchCV for improved performance.

Tools and Libraries

    Programming Language: Python

    Libraries:

        Data manipulation: pandas, numpy

        Modeling: scikit-learn, skopt

        Logging: logging

Results

    The model achieved accurate predictions for Pclass with the following steps:

    Preprocessing effectively reduced noise and improved data quality.

    Hyperparameter tuning further optimized model performance.


Author

Rohit Singh Rawat

Data Analyst and Machine Learning Enthusiast