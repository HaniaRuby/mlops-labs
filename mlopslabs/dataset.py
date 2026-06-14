import os
import dagshub
import hydra
import joblib
import mlflow
from mlopslabs.modeling import train
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def run_pipeline(cfg: DictConfig):
    # 1. Load Data
    raw_data = pd.read_csv(cfg.data.raw_path)

    # 2. Features/Target from Hydra
    features = list(cfg.data.num_features) + list(cfg.data.cat_features)
    X = raw_data[features]
    y = raw_data[cfg.data.target]

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.train_params.test_size,
        random_state=cfg.train_params.random_state,
    )

    # DagsHub & MLflow initialization
    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(
        repo_owner="haniaruby",
        repo_name=cfg.mlflow.repo_name,
        mlflow=cfg.mlflow.use_mlflow,
    )
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment("titanic-survival-prediction")

    print(f"Training {cfg.model.name}...")

    # ONE single run block context for everything cloud-related
    with mlflow.start_run(run_name=f"run-{cfg.model.name}"):
        # Log basic Hydra configurations metadata
        mlflow.log_param("model_type", cfg.model.name)
        for param_name, param_val in cfg.model.params.items():
            mlflow.log_param(param_name, param_val)

        # 4. Train
        model_pipe = train.build_and_train(X_train, y_train, cfg)

        # 5. Evaluate
        score = model_pipe.score(X_test, y_test)
        print(f"Model Score ({cfg.model.name}): {score:.4f}")

        # Log metric to the SAME run
        mlflow.log_metric("accuracy", score)

        # Log and register the in-memory object directly to DagsHub
        print("Uploading model artifact directly to DagsHub MLflow Registry...")
        mlflow.sklearn.log_model(
            sk_model=model_pipe,  # The in-memory scikit-learn object
            artifact_path="model_artifact",  # Unique cloud-path directory
            registered_model_name="TitanicSurvivalModel",
        )

    # 6. Save locally for DVC tracking pipeline consistency
    os.makedirs("models", exist_ok=True)
    save_path = "models/model.pkl"
    joblib.dump(model_pipe, save_path)
    print(f"Local DVC artifact backup saved to {save_path}")


if __name__ == "__main__":
    run_pipeline()