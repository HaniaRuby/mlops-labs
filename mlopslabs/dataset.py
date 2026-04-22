import hydra
import joblib
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split

from mlopslabs.modeling import train


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def run_pipeline(cfg: DictConfig):  # 1. Accept the config object
    # 2. Use cfg for paths instead of the old config.py
    raw_data = pd.read_csv(cfg.data.raw_path)

    # These can still come from a config or be moved to YAML too
    features = cfg.data.num_features + cfg.data.cat_features
    X = raw_data[features]
    y = raw_data[cfg.data.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.model.test_size, random_state=cfg.model.random_state
    )

    # 3. Use Hydra to decide which model to train
    print(f"Training {cfg.model.name}...")
    model_pipe = train.build_and_train(X_train, y_train, cfg)

    score = model_pipe.score(X_test, y_test)
    print(f"Model Score ({cfg.model.name}): {score:.4f}")

    # 4. Save using the path from YAML
    joblib.dump(model_pipe, cfg.data.processed_path)
    print(f"Model saved to {cfg.data.processed_path}")


if __name__ == "__main__":
    run_pipeline()
