import joblib
import os
def save_model(model, scaler, model_path="Models/model.pkl", scaler_path="Models/scaler.pkl"):
    os.makedirs(os.path.dirname(model_path),exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print("model Saved successfully")

def load_model(model_path="Models/model.pkl", scaler_path="Models/scaler.pkl"):
    model=joblib.load(model_path)
    scaler=joblib.load(scaler_path)
    print("model loaded successfully")
    return model, scaler