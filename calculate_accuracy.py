import numpy as np
from model import classify


def calculate_errors(y_true, y_pred):
 
    return np.sum(y_true != y_pred)


def calculate_accuracy(y_true, y_pred):
   
    return np.mean(y_true == y_pred)


def evaluate_model(df_train, df_test, features_design, theta_optimal, model):
    """
    Calculates errors and accuracy for train and test data.

    Parameters:
    - df_train: pandas DataFrame
    - df_test: pandas DataFrame
    - features_design: function for features preparation
    - theta_optimal: numpy array - optimal parameters
    - model: log regression
    
    """
    X_train, X_test = df_train.drop("Survived", axis=1).to_numpy(), df_test.drop("Survived", axis=1).to_numpy()
    y_train, y_test = df_train["Survived"].to_numpy(), df_test["Survived"].to_numpy()

    features_matrix_train = features_design(X_train)
    features_matrix_test = features_design(X_test)

    y_pred_train = classify(theta_optimal, features_matrix_train, model)
    y_pred_test = classify(theta_optimal, features_matrix_test, model)


    train_accuracy = calculate_accuracy(y_train, y_pred_train)
    test_accuracy = calculate_accuracy(y_test, y_pred_test)

    return  train_accuracy, test_accuracy
