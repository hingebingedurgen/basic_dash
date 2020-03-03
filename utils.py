from sklearn.datasets import (
    load_iris,
    load_digits,
    load_wine,
    load_breast_cancer
    )

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def load_data(dataset):
    if dataset == 'Iris':
        return load_iris()
    elif dataset == 'Wine':
        return load_wine()
    return load_breast_cancer()


def load_model(model):
    if model == 'Logistic Regression':
        return LogisticRegression()
    elif model == 'Random Forest':
        return RandomForestClassifier()
    return SVC()
