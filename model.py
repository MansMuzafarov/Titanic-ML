import numpy as np


def logistic_func(theta, feature_matrix):

    arg = np.dot(feature_matrix, theta)
    arg = np.where(arg > 18, 18, np.where(arg < -18, -18, arg))

    return 1.0 / (1 + np.exp(-arg))


def classify(theta, features_matrix, model):

    model_results = model(theta, features_matrix)

    classifications = np.where(model_results > 0.5, 1, 0)

    return classifications



# Loss-function and its derrivative: 

def log_likelihood(theta, feature_matrix, y_train, model, lambda_coef):

    norm_theta = np.linalg.norm(theta)
    if norm_theta > 1e6:
        print(f"Warning: Theta norm too large: {norm_theta}. Regularizing heavily.")
        norm_theta = 1e6

    epsilon = 1e-10
    y_model_predictions = model(theta, feature_matrix)
    result = (
        np.sum(
            y_train * np.log(y_model_predictions + epsilon)
            + (1 - y_train) * np.log(1 - y_model_predictions + epsilon)
        )
        - (lambda_coef / 2) * (np.linalg.norm(theta)) ** 2
    )

    return result


def negative_log_likelihood(theta, feature_matrix, y_train, model, lambda_coef):

    return -1 * log_likelihood(theta, feature_matrix, y_train, model, lambda_coef)


def log_likelihood_derivative(theta, feature_matrix, y_train, model, lambda_coef):

    y_model_predictions = model(theta, feature_matrix)

    delta_y = y_train - y_model_predictions

    result = np.dot(feature_matrix.T, delta_y) - lambda_coef * theta

    assert result.shape == theta.shape

    return result


def negative_log_likelihood_derivative(
    theta, feature_matrix, y_train, model, lambda_coef
):

    return -1 * log_likelihood_derivative(
        theta, feature_matrix, y_train, model, lambda_coef
    )