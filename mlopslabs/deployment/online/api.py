import joblib
import litserve as ls
import pandas as pd
from pydantic import ValidationError

from mlopslabs.deployment.online.requests import InferenceRequest


class InferenceAPI(ls.LitAPI):

    def setup(self, device="cpu"):
        # Load your local trained scikit-learn pipeline
        model_path = "models/model.pkl"
        print(f"Loading local deployment artifact from {model_path}...")
        self._model = joblib.load(model_path)

    def decode_request(self, request):
        """Validates input payload structure using Pydantic contracts and converts

        to a clean evaluation dataframe matching your feature pipelines.
        """
        try:
            # Enforce validation using our batch model contract
            validated_data = InferenceRequest(**request)

            # Extract the records array and convert into a clean Pandas DataFrame
            # This perfectly preserves the feature name keys for your pipeline transformers
            records_dict = [
                record.model_dump() for record in validated_data.input
            ]
            df_features = pd.DataFrame(records_dict)
            return df_features

        except (ValidationError, KeyError, TypeError) as err:
            print(f"❌ API Request Validation Failure: {err}")
            return None

    def predict(self, x_df):
        """Feeds the batch dataframe directly through the model pipeline execution layers."""
        if x_df is not None:
            predictions = self._model.predict(x_df)
            probabilities = self._model.predict_proba(x_df)
            return {"preds": predictions, "probs": probabilities}
        return None

    def encode_response(self, output):
        """Formats the internal predictions array into a professional JSON array output."""
        if output is None:
            return {"message": "Error Occurred", "results": []}

        response_array = []
        # Zip predictions and probabilities array to generate organized records
        for pred, prob in zip(output["preds"], output["probs"]):
            response_array.append(
                {
                    "prediction": int(pred),
                    "label": "Survived" if pred == 1 else "Deceased",
                    "confidence": {
                        "deceased_prob": round(float(prob[0]), 4),
                        "survived_prob": round(float(prob[1]), 4),
                    },
                }
            )

        return {
            "message": "Response Produced Successfully",
            "results": response_array,
        }