import pandas as pd

from . import config


def preprocess_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and encodes Titanic data."""
    # Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Select features and encode
    X = pd.get_dummies(df[config.FEATURES], drop_first=True)

    if config.TARGET in df.columns:
        return pd.concat([X, df[config.TARGET]], axis=1)
    return X
