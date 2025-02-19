import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import normalize_data
from model import logistic_func, classify, negative_log_likelihood, negative_log_likelihood_derivative
from calculate_accuracy import calculate_errors


def batch_gradient_descent(
    batch_size,
    feature_matrix_train,
    y_train,
    init_theta,
    alpha,
    nIter,
    model,
    lambda_coef,
):

    epsilon = 1e-6
    n_samples = feature_matrix_train.shape[0]
    theta_iter = init_theta.copy()
    negative_log_likelihood_function_values = []

    for iteration in range(nIter):

        negative_log_likelihood_function_values.append(
            negative_log_likelihood(
                theta=theta_iter,
                feature_matrix=feature_matrix_train,
                y_train=y_train,
                model=logistic_func,
                lambda_coef=lambda_coef,
            )
        )

        current_alpha = alpha / (1 + iteration / nIter)

        # Choosing the random batch from doata:

        indices = np.random.choice(n_samples, size=batch_size, replace=False)

        # Extracting the features and labels of objects in the batch:

        batch_features = feature_matrix_train[indices]

        batch_labels = y_train[indices]

        grad = negative_log_likelihood_derivative(
            theta=theta_iter,
            feature_matrix=batch_features,
            y_train=batch_labels,
            model=model,
            lambda_coef=lambda_coef,
        )

        if (
            iteration > 1
            and abs(
                negative_log_likelihood_function_values[-1]
                - negative_log_likelihood_function_values[-2]
            )
            < epsilon
            and np.linalg.norm(grad) < epsilon
        ):
            print(f"Converged after {iteration} iterations with delta < {epsilon}")
            break

        theta_iter = theta_iter - current_alpha * grad

    plt.plot(negative_log_likelihood_function_values)
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.title("Loss function in batch gradient method")
    plt.savefig("Loss_function_in_Batch_gradient_method_.png")
    plt.close()

    return theta_iter


# Newton's method:


def newton_method(feature_matrix_train, y_train, init_theta, nIter, model, lambda_coef):
    epsilon = 1e-6
    theta_iter = init_theta.copy()
    negative_log_likelihood_values = []

    for iteration in range(nIter):
        # Model predictions
        y_model = model(theta_iter, feature_matrix_train)

        # Gradient for -log-likelihood:
        gradient = negative_log_likelihood_derivative(
            theta=theta_iter,
            feature_matrix=feature_matrix_train,
            y_train=y_train,
            model=logistic_func,
            lambda_coef=lambda_coef,
        )

        # Hessian:
        R = np.diag(y_model * (1 - y_model))
        hessian = np.dot(
            feature_matrix_train.T, np.dot(R, feature_matrix_train)
        ) + lambda_coef * np.eye(feature_matrix_train.shape[1])

        # Parameters update
        try:
            hessian_inv = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            print("Error, Hessian matrix couldn't be inverted")
            break

        delta_theta = np.dot(hessian_inv, gradient)
        theta_iter -= delta_theta  # Update step for minimization

        # Compute -log-likelihood and save it
        negative_log_likelihood_values.append(
            negative_log_likelihood(
                theta_iter, feature_matrix_train, y_train, model, lambda_coef
            )
        )

        # Convergence check
        if np.linalg.norm(delta_theta) < epsilon:
            print(f"Newton method converged on iteration number: {iteration + 1}")
            break

    # Plot -log-likelihood
    plt.plot(negative_log_likelihood_values)
    plt.xlabel("Iteration")
    plt.ylabel("-Log-likelihood")
    plt.title("Loss function in Newton's method")
    plt.savefig("Loss_function_in_Newton's_method.png")
    plt.close()

    return theta_iter


def cross_validation_algorithm(
    df,
    number_of_folds,
    lambda_coef_values,
    batch_size,
    alpha,
    nIter,
    features_design_function,
    features_to_normalize,
    model,
):

    size_of_data_i = df.shape[0] // number_of_folds

    average_number_of_errors_on_test_data_values = np.array([])

    for lambda_coef_value in lambda_coef_values:

        test_errors = np.array([])

        for i in range(number_of_folds):

            # Split into training and testing data for the current fold
            start_index_i = i * size_of_data_i
            stop_index_i = (i + 1) * size_of_data_i
            test_indices = range(start_index_i, stop_index_i)

            df_test_i = df.iloc[test_indices]
            df_train_i = df.drop(df.index[test_indices])

            df_train_i_normalized, means_i, sigmas_i = normalize_data(
                df=df_train_i, features_to_normalize=features_to_normalize
            )

            df_test_i_normalized = df_test_i.copy()
            df_test_i_normalized[features_to_normalize] = (
                df_test_i[features_to_normalize] - means_i
            ) / sigmas_i

            # Convert to numpy arrays
            X_train_i_normalized = df_train_i_normalized.drop(
                "Survived", axis=1
            ).to_numpy()
            y_train_i = df_train_i_normalized["Survived"].to_numpy()

            X_test_i_normalized = df_test_i_normalized.drop(
                "Survived", axis=1
            ).to_numpy()
            y_test_i = df_test_i_normalized["Survived"].to_numpy()

            # Feature transformation for train dataset (numpy matrix) and test dataset:

            feature_matrix_train_i_normalized = features_design_function(
                X=X_train_i_normalized
            )
            feature_matrix_test_i_normalized = features_design_function(
                X=X_test_i_normalized
            )

            # Initialize theta
            init_theta_i = np.zeros(feature_matrix_train_i_normalized.shape[1])

            # Optimize theta using gradient descent
            theta_optimal_i = batch_gradient_descent(
                batch_size=batch_size,
                feature_matrix_train=feature_matrix_train_i_normalized,
                y_train=y_train_i,
                init_theta=init_theta_i,
                alpha=alpha,
                nIter=nIter,
                model=logistic_func,
                lambda_coef=lambda_coef_value,
            )

            # Calculate test errors

            test_error_value = test_error_value = calculate_errors( y_test_i, classify(theta_optimal_i, feature_matrix_test_i_normalized, model))


            test_errors = np.append(test_errors, test_error_value)

        # Average errors for the current lambda
        average_error = np.mean(test_errors)
        average_number_of_errors_on_test_data_values = np.append(
            average_number_of_errors_on_test_data_values, average_error
        )

    plt.plot(
        lambda_coef_values,
        average_number_of_errors_on_test_data_values,
        color="r",
        label="Test Data",
    )
    plt.title("Number of errors as a function of lambda")
    plt.xlabel("Lambda coefficient")
    plt.ylabel("Number of errors")
    plt.legend()
    plt.savefig(
        "Average number of errors on test data (crossvalidation) on as a function of lambda.png"
    )
    plt.close()

    best_lambda = lambda_coef_values[
        np.argmin(average_number_of_errors_on_test_data_values)
    ]

    return best_lambda


# Generalization of optimization methods:


def optimize(
    method,
    feature_matrix_train,
    y_train,
    init_theta,
    alpha,
    nIter,
    lambda_coef,
    model,
    batch_size=None,
):

    if method == "batch_gradient_descent":
        return batch_gradient_descent(
            batch_size=batch_size,
            feature_matrix_train=feature_matrix_train,
            y_train=y_train,
            init_theta=init_theta,
            alpha=alpha,
            nIter=nIter,
            model=model,
            lambda_coef=lambda_coef,
        )

    elif method == "newton_method":
        return newton_method(
            feature_matrix_train=feature_matrix_train,
            y_train=y_train,
            init_theta=init_theta,
            nIter=nIter,
            model=model,
            lambda_coef=lambda_coef,
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")


# Optimization pipeline:


def optimize_with_pipeline(
    methods_sequence,
    df_train,
    features_to_normalize,
    features_design_function,
    number_of_folds,
    alpha,
    nIter,
    lambda_coef_values,
    model,
    batch_size,
):

    print("Data preparation processing...")

    # Step 1: Normalize features

    df_train_normalized, means, sigmas = normalize_data(
        df=df_train, features_to_normalize=features_to_normalize
    )

    X_train_normalized = df_train_normalized.drop(columns="Survived").to_numpy()

    y_train = df_train["Survived"].to_numpy()

    features_matrix_train_normalized = features_design_function(X=X_train_normalized)

    print("Cross-validation starts...")
    # Step 2: Cross-validation to find the best lambda
    best_lambda = cross_validation_algorithm(
        df=df_train,
        number_of_folds=number_of_folds,
        lambda_coef_values=lambda_coef_values,
        batch_size=batch_size,
        alpha=alpha,
        nIter=nIter,
        features_design_function=features_design_function,
        features_to_normalize=features_to_normalize,
        model=model,
    )
    print("Cross-validation is finished.")
    print(f"Best lambda: {best_lambda}")

    # Step 3: Initialize theta
    n_features = features_matrix_train_normalized.shape[1]
    init_theta = np.zeros(n_features)

    theta = init_theta
    for method in methods_sequence:
        print(f"Applying method: {method}")
        theta = optimize(
            method=method,
            feature_matrix_train=features_matrix_train_normalized,
            y_train=y_train,
            init_theta=theta,
            alpha=alpha,
            nIter=nIter,
            lambda_coef=best_lambda,
            model=model,
            batch_size=batch_size,
        )

    print("Optimization complete.")
    feature_names = list(df_train.columns)

    # theta = denormalize_theta(theta_normalized = theta, means = means, sigmas = sigmas, features_to_normalize = features_to_normalize, feature_names = feature_names)

    return theta