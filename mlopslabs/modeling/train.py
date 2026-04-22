from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlopslabs import config


def build_and_train(X_train, y_train, cfg):

    if cfg.model.name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=cfg.model.n_estimators, random_state=cfg.model.random_state
        )
    else:
        clf = LogisticRegression(max_iter=1000)

    # 1. Preprocessing for numerical data
    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    # 2. Preprocessing for categorical data
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 3. Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, config.NUM_FEATURES),
            ("cat", cat_transformer, config.CAT_FEATURES),
        ]
    )

    # 5. Create the full Pipeline
    model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

    # 6. Train
    model_pipeline.fit(X_train, y_train)
    return model_pipeline
