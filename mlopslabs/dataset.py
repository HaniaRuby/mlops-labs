import hydra
import joblib
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split
import os

from mlopslabs.modeling import train

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
        X, y, 
        test_size=cfg.train_params.test_size, 
        random_state=cfg.model.params.random_state
    )

    # 4. Train
    print(f"Training {cfg.model.name}...")
    model_pipe = train.build_and_train(X_train, y_train, cfg)

    # 5. Evaluate
    score = model_pipe.score(X_test, y_test)
    print(f"Model Score ({cfg.model.name}): {score:.4f}")

    # 6. Save (Ensure this matches your DVC -o flag!)
    # I recommend using models/model.pkl to match your DVC stage
    os.makedirs("models", exist_ok=True)
    save_path = "models/model.pkl" 
    joblib.dump(model_pipe, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    run_pipeline()