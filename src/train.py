import pandas as pd
import yaml
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load parameters
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

test_size = params["train"]["test_size"]
random_state = params["train"]["random_state"]

n_estimators = params["model"]["n_estimators"]
max_depth = params["model"]["max_depth"]

# Load processed data
df = pd.read_csv("data/processed.csv")

X = df.drop(["target", "split"], axis=1)
y = df["target"]

# Use the same split as preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=random_state
)

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/model.pkl")

print("Model training completed successfully!")
print(f"n_estimators: {n_estimators}")
print(f"max_depth: {max_depth}")
print("Model saved to models/model.pkl")