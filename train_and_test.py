import numpy as np
import pandas as pd
from data_preprocessing import prepare_titanic_data, normalize_data, train_test_split
from features_designs import feature_affine_design
from optimization import optimize_with_pipeline
from model import logistic_func, classify
from calculate_metrics import evaluate_model, calculate_accuracy




def train_and_test():
    
    #Data loading: 

    df = pd.read_csv("Data/train.csv")  # our data

    # Let's prepare our data for the further manipulations:

    # Here we should decide if we are going to use information we could extract from the 'Name' feature:

    use_name_features = True

    df = prepare_titanic_data(df=df, use_name_features=use_name_features)

    # Data split:

    test_size_ratio = 0.2

    df_train, df_test = train_test_split(
        df=df, test_size_ratio=test_size_ratio, random_seed=42
    )

   
    numerical_features = ["NameLength", "Age", "FamilySize", "LogFare"]

    if use_name_features == False:

        numerical_features = ["Age", "FamilySize", "LogFare"]


    # Normalization of numerical features and mean values and standard deviations of the selected features (numerical features: NameLength, Age, FamilySize, LogFare):

    features_to_normalize = numerical_features.copy()

    # Optimization:

    methods_sequence = ["batch_gradient_descent", "newton_method"]

    number_of_folds = 5

    batch_fraction = 0.2
    batch_size = int(batch_fraction * df_train.shape[0])

    alpha = 0.001

    nIter = 10000

    lambda_coef_values = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])

    model = logistic_func

    theta_optimal_normalized = optimize_with_pipeline(
        methods_sequence=methods_sequence,
        df_train=df_train,
        features_to_normalize=features_to_normalize,
        features_design_function=feature_affine_design,
        number_of_folds=number_of_folds,
        alpha=alpha,
        nIter=nIter,
        lambda_coef_values=lambda_coef_values,
        model=model,
        batch_size=batch_size,
    )

    # Data normalization:

    df_train_normalized, means, sigmas = normalize_data(
        df=df_train, features_to_normalize=features_to_normalize
    )

    df_test_normalized = df_test.copy()

    df_test_normalized[features_to_normalize] = (
        df_test[features_to_normalize] - means
    ) / sigmas

    train_accuracy, test_accuracy, precision, recall = evaluate_model(
        df_train=df_train_normalized,
        df_test=df_test_normalized,
        features_design=feature_affine_design,
        theta_optimal=theta_optimal_normalized,
        model=model,
    )



    print("Train accuracy: ", np.round(train_accuracy, 2))
    print("Test accuracy: ", np.round(test_accuracy, 2), "\n")
    print("Precision: ", np.round(precision, 2), "\n")
    print("Recall: ", np.round(recall, 2), "\n")


    
# Final predictions for Kaggle's test data:


def make_predictions_for_kaggle():
    
    print("Predictions for kaggle test data process has started ...")
    
    df_train = pd.read_csv("Data/train.csv")
    
    df_test = pd.read_csv("Data/test.csv")
    
    # Here we save PassengerIds for test-data: we are going to use it in a final stage
    
    passenger_ids = df_test["PassengerId"].copy()
    
    df_train = prepare_titanic_data(df = df_train, is_test_data = False)
    
    title_columns = [col for col in df_train.columns if col.startswith("Title_")]
    
    df_test = prepare_titanic_data(df = df_test, is_test_data = True)
    
    for col in title_columns:
        if col not in df_test.columns:
           df_test[col] = 0
    
    df_test = df_test[df_train.columns.drop("Survived", errors="ignore")]
    
    use_name_features = True
    
    numerical_features = ["NameLength", "Age", "FamilySize", "LogFare"]

    if use_name_features == False:

        numerical_features = ["Age", "FamilySize", "LogFare"]


    # Normalization of numerical features and mean values and standard deviations of the selected features (numerical features: NameLength, Age, FamilySize, LogFare):

    features_to_normalize = numerical_features.copy()

    # Optimization:

    methods_sequence = ["batch_gradient_descent", "newton_method"]

    number_of_folds = 5

    batch_fraction = 0.2
    batch_size = int(batch_fraction * df_train.shape[0])

    alpha = 0.001

    nIter = 10000

    lambda_coef_values = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])

    model = logistic_func

    theta_optimal_normalized = optimize_with_pipeline(
        methods_sequence=methods_sequence,
        df_train=df_train,
        features_to_normalize=features_to_normalize,
        features_design_function=feature_affine_design,
        number_of_folds=number_of_folds,
        alpha=alpha,
        nIter=nIter,
        lambda_coef_values=lambda_coef_values,
        model=model,
        batch_size=batch_size,
    )

    # Data normalization:

    df_train_normalized, means, sigmas = normalize_data(
        df=df_train, features_to_normalize=features_to_normalize
    )
    
    df_test_normalized = df_test.copy()

    df_test_normalized[features_to_normalize] = (
        df_test[features_to_normalize] - means
    ) / sigmas
    
    X_train_normalized = df_train_normalized.drop("Survived", axis=1).to_numpy()

    X_test_normalized = df_test_normalized.to_numpy()  
    
    features_matrix_train_normalized = feature_affine_design(X_train_normalized)

    feature_matrix_test_normalized = feature_affine_design(X_test_normalized)
    
    # Predictions:
    
    y_pred_train = classify(theta = theta_optimal_normalized, features_matrix = features_matrix_train_normalized, model = logistic_func)
    
    y_pred_test = classify(theta = theta_optimal_normalized, features_matrix = feature_matrix_test_normalized, model = logistic_func)
    
    # Accuracy for train data:
    
    train_accuracy = calculate_accuracy(y_true = df_train["Survived"].to_numpy(), y_pred = y_pred_train)
    
    
    print("Train accuracy: ", np.round(train_accuracy, 2))
    
    df_submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": y_pred_test})
    df_submission.to_csv("submission.csv", index=False)
    print("Predictions were saved to submission.csv file!")
    
    