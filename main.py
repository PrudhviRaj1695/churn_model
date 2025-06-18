from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Insurance Churn Predictor", version="1.0")

# Load model
# Use a model path relative to this file to avoid relying on a hard-coded
# absolute path which will not exist in other environments.
import os
model_path = os.path.join(os.path.dirname(__file__), "Churn_Model_Pickle.pkl")
model = joblib.load(model_path)


# Define input structure
class ChurnInput(BaseModel):
    Category_Premium: int
    Claim_Amount: int
    Claim_Reason: str
    Data_confidentiality: str

@app.post("/predict/")
def predict_churn(input_data: ChurnInput):
    try:
        # Feature engineering
        claim_to_premium_ratio = input_data.Claim_Amount / (input_data.Category_Premium + 1)
        premium_claim_interaction = input_data.Category_Premium * input_data.Claim_Amount
        high_claim_flag = int(input_data.Claim_Amount > 10000)

        # Base row
        data = {
            "Claim Amount": input_data.Claim_Amount,
            "Category Premium": input_data.Category_Premium,
            "Premium/Amount Ratio": input_data.Category_Premium / (input_data.Claim_Amount + 1),
            "claim_to_premium_ratio": claim_to_premium_ratio,
            "premium_claim_interaction": premium_claim_interaction,
            "high_claim_flag": high_claim_flag,
            "Claim Reason_" + input_data.Claim_Reason: 1,
            "Data confidentiality_" + input_data.Data_confidentiality: 1,
        }

        # Full feature list used during training
        expected_cols = [
            'Claim Amount', 'Category Premium', 'Premium/Amount Ratio',
            'claim_to_premium_ratio', 'premium_claim_interaction', 'high_claim_flag',
            'Claim Reason_Other', 'Claim Reason_Phone', 'Claim Reason_Travel',
            'Data confidentiality_Low', 'Data confidentiality_Medium', 'Data confidentiality_Very low'
        ]

        # Fill missing columns with 0
        model_input = {col: data.get(col, 0) for col in expected_cols}
        df_input = pd.DataFrame([model_input])

        # Predict
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        return {
            "churn_prediction": float(prediction),
            "probability": round(probability, 4)
        }

    except Exception as e:
        return {"detail": f"Prediction error: {str(e)}"}
