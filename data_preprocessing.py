import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def train_test_split(df, test_size_ratio, random_seed=None):

    if random_seed is not None:
        np.random.seed(random_seed)

    indices = list(range(len(df)))
    np.random.shuffle(indices)
    test_indices = indices[: int(test_size_ratio * len(df))]
    df_test = df.iloc[test_indices]
    df_train = df.drop(index=df.index[test_indices])

    return df_train, df_test


def normalize_data(df, features_to_normalize):

    means = df[features_to_normalize].mean()
    sigmas = df[features_to_normalize].std()

    # Normalization of the selected features:
    df_normalized = df.copy()
    df_normalized[features_to_normalize] = (df[features_to_normalize] - means) / sigmas

    return df_normalized, means, sigmas


def prepare_titanic_data(df, use_name_features=True, is_test_data = False):

    df = df.copy()
    
    # Here we add fictial feature for test_data from Kaggle: 
    
    if is_test_data:
        df["Survived"] = -1

    # PassengerId - to drop:

    df = df.drop(columns="PassengerId")

    # Pclass - one hot encoding:

    df["Pclass_2"] = (df["Pclass"] == 2).astype(int)
    df["Pclass_3"] = (df["Pclass"] == 3).astype(int)

    df = df.drop(columns="Pclass")

    # 'Sex' - label encoding

    df["Sex"] = df["Sex"].map({"female": 1, "male": 0})

    # 'SibSp' and 'Parch' - we create a new feature - size of the family

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    df = df.drop(columns=["SibSp", "Parch"])

    # Ticket: here I have decided just to delete this feature from the dataset

    df = df.drop(columns="Ticket")

    # Fare: Here I have decided to compute log(Fare + 1) to reduce the impact of small variations

    df["LogFare"] = np.log1p(df["Fare"])  # np.log1p(x) = log(x + 1)

    df = df.drop(columns="Fare")

    # Cabin: we could notice that there are a lot of 'NaN' values in this column, but also that cabine name have people from the first class. So, I have decided to create a new feature "HasCabin"

    df["HasCabin"] = df["Cabin"].notnull().astype(int)
    df = df.drop(columns="Cabin")

    # Embarked: one hot encoding

    # print(df[df['Embarked'].isnull()])
    # print(df['Embarked'].isnull().sum())                                #only two gaps, both are passengers from the first class

    frac_1_st_class_from_S = (
        df[(df["Pclass_2"] == 0) & (df["Pclass_3"] == 0)]["Embarked"] == "S"
    ).astype(int).to_numpy().sum() / (df["Embarked"] == "S").astype(
        int
    ).to_numpy().sum()

    frac_1_st_class_from_C = (
        df[(df["Pclass_2"] == 0) & (df["Pclass_3"] == 0)]["Embarked"] == "C"
    ).astype(int).to_numpy().sum() / (df["Embarked"] == "C").astype(
        int
    ).to_numpy().sum()

    frac_1_st_class_from_Q = (
        df[(df["Pclass_2"] == 0) & (df["Pclass_3"] == 0)]["Embarked"] == "Q"
    ).astype(int).to_numpy().sum() / (df["Embarked"] == "Q").astype(
        int
    ).to_numpy().sum()

    Ports = np.array(["S", "C", "Q"], dtype=str)
    fracs = np.array(
        [frac_1_st_class_from_S, frac_1_st_class_from_C, frac_1_st_class_from_Q]
    )
    most_common_port_for_the_1_st_class = Ports[np.where(fracs == fracs.max())]

    most_common_port_for_the_1_st_class = str(most_common_port_for_the_1_st_class[0])

    df["Embarked"] = df["Embarked"].fillna(most_common_port_for_the_1_st_class)

    # one hot encoding for the 'Embarked' feature:

    df = pd.get_dummies(df, columns=["Embarked"], prefix="Embarked", drop_first=True)

    if use_name_features == True:

        df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

        df["NameLength"] = df["Name"].apply(len)

        df = df.drop(columns="Name")

        df = df[
            ["Survived", "Pclass_2", "Pclass_3", "Title", "NameLength"]
            + [
                col
                for col in df.columns
                if col
                not in ["Survived", "Pclass_2", "Pclass_3", "Title", "NameLength"]
            ]
        ]

        # Titles grouping:
        df["Title"] = df["Title"].replace(["Mme", "Ms", "Lady", "Dona"], "Mrs")
        df["Title"] = df["Title"].replace(
            ["Capt", "Col", "Major", "Dr", "Rev"], "Officer"
        )
        df["Title"] = df["Title"].replace(
            ["Jonkheer", "Don", "Sir", "the Countess"], "Noble"
        )
        df["Title"] = df["Title"].replace(["Mlle"], "Miss")
        df["Title"] = df["Title"].replace(["Countess"], "Noble")

        # print(df['Title'].unique())

        # print(df['Title'].value_counts())

        # print(df.groupby('Title')['Survived'].mean())

        # We use one-hot encoding for titles:

        df = pd.get_dummies(df, columns=["Title"], prefix="Title", drop_first=True)
        
        title_columns = [col for col in df.columns if col.startswith("Title_")]
        

        # correlation = df['NameLength'].corr(df['Survived'])
        # print(f"Correlation between Name Length and Survived: {correlation:.4f}")            # correlation coef here is about 0.33 - weak positive correlation

        # 'Age': here we have some null values

        # print(df['Age'].isnull().sum())             177 nulls

        # here we simply fill these empty spaces in 'Age' by medians, grouped by title

        for title_col in title_columns:

            median_age = df.loc[df[title_col] == 1, "Age"].median()

            df.loc[(df[title_col] == 1) & (df["Age"].isnull()), "Age"] = median_age

        # Separately for master's title: (because this title is encoded as zeros in the rest of the titles)

        master_mask = (df[title_columns].sum(axis=1)== 0)

        master_median_age = df.loc[master_mask, "Age"].median()

        df.loc[master_mask & df["Age"].isnull(), "Age"] = master_median_age

        # Final stage of data analysis: In case with 'Name' extracted information we have 15 features: 4 - numerical features, 11 - categorical.
        # Let's have some new order of these features in out dataframe.
        # In case where we don't use 'Name' information we have 9 features: 3 - numerical, 6 - categorical.

        numeric_features = ["NameLength", "Age", "FamilySize", "LogFare"]

        remaining_features = [
            col
            for col in df.columns
            if col not in numeric_features and col != "Survived"
        ]

        new_column_order = ["Survived"] + numeric_features + remaining_features
        df = df[new_column_order]

    else:

        df = df.drop(columns="Name")
        # For the dataset without 'Name' information we fill 'Age' just by it's median:

        df["Age"] = df["Age"].fillna(df["Age"].median())

        numeric_features = ["Age", "FamilySize", "LogFare"]

        remaining_features = [
            col
            for col in df.columns
            if col not in numeric_features and col != "Survived"
        ]

        new_column_order = ["Survived"] + numeric_features + remaining_features
        df = df[new_column_order]
    
    # For test_data: let's now delete fictial feature:
    
    if is_test_data:
        df = df.drop(columns=["Survived"])
    
    with open("output.txt", "w") as f:
        f.write(df.to_string(index=True))

    # print(df['Survived'].value_counts(normalize=True))

    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig("Correlation_matrix.png")
    plt.close()

    return df
