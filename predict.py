import os
import dagshub
import mlflow
import pandas as pd

# 1. Authenticate with the environment token you exported
dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
mlflow.set_tracking_uri("https://dagshub.com/haniaruby/mlops-labs.mlflow")

# 2. Tell MLflow to grab the specific model tagged as 'Production'
model_uri = "models:/TitanicSurvivalModel/1"
print(f"Loading remote model from: {model_uri}")
model = mlflow.sklearn.load_model(model_uri)

# 3. Create a synthetic test passenger payload to verify it functions
# (Ensure these column keys match your local data structure config exactly!)
sample_passenger = pd.DataFrame(
    [
        {
            "Pclass": 3,
            "Age": 22.0,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 7.25,
            "Sex": "male",
            "Embarked": "S",
        }
    ]
)

# 4. Infer!
prediction = model.predict(sample_passenger)
probability = model.predict_proba(sample_passenger)

print("\n=== INFERENCE RESULT ===")
print(f"Predicted Class: {prediction[0]} (0 = Deceased, 1 = Survived)")
print(f"Confidence Profile [Death Prob, Survival Prob]: {probability[0]}")