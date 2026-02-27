import os
import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run_training():
    # 1. Load dataset
    diabetes = load_diabetes()
    X = diabetes.data   # shape (442, 10)
    y = diabetes.target # continuous values

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train a model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 4. Quick eval just so we know it learned
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model MSE: {mse:.3f}, R2: {r2:.3f}")

    # 5. Make sure model/ directory exists
    os.makedirs("model", exist_ok=True)

    # 6. Save model to model/model.pkl
    joblib.dump(model, "model/model.pkl")
    print("Saved trained model to model/model.pkl")

if __name__ == "__main__":
    run_training()