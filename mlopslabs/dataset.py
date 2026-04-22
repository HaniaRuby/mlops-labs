import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from mlopslabs import config
from mlopslabs.modeling import train  # Ensure folder is 'modeling'


def run_pipeline():
    # Load Data
    raw_data = pd.read_csv(config.RAW_DATA_DIR / "train.csv")
    X = raw_data[config.NUM_FEATURES + config.CAT_FEATURES]
    y = raw_data[config.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train 2 models and compare
    print("Training Logistic Regression...")
    lr_pipe = train.build_and_train(X_train, y_train, model_type="lr")
    lr_score = lr_pipe.score(X_test, y_test)

    print("Training Random Forest...")
    rf_pipe = train.build_and_train(X_train, y_train, model_type="rf")
    rf_score = rf_pipe.score(X_test, y_test)

    print(f"LR Score: {lr_score:.4f} | RF Score: {rf_score:.4f}")

    # Save the winner
    winner = rf_pipe if rf_score > lr_score else lr_pipe
    joblib.dump(winner, config.MODELS_DIR / "model.pkl")
    print(f"Best model saved to {config.MODELS_DIR / 'model.pkl'}")


if __name__ == "__main__":
    run_pipeline()
