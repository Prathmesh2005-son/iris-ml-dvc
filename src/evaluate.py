import json
import pandas as pd
import yaml
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load parameters
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

test_size = params["train"]["test_size"]
random_state = params["train"]["random_state"]

# Load processed data
df = pd.read_csv("data/processed.csv")

X = df.drop(["target", "split"], axis=1)
y = df["target"]

# Recreate the same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
)

# Load trained model
model = joblib.load("models/model.pkl")

# Make predictions
predictions = model.predict(X_test)

# Calculate metrics
metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "precision": precision_score(
        y_test,
        predictions,
        average="weighted"
    ),
    "recall": recall_score(
        y_test,
        predictions,
        average="weighted"
    ),
    "f1_score": f1_score(
        y_test,
        predictions,
        average="weighted"
    )
}

# Save metrics
with open("metrics/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Evaluation completed successfully!")
print(json.dumps(metrics, indent=4))
print("Metrics saved to metrics/metrics.json")