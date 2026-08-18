import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/dataset.csv"
OUTPUT_FILE = "data/processed.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Combine train and test data
train_data = X_train.copy()
train_data["target"] = y_train
train_data["split"] = "train"

test_data = X_test.copy()
test_data["target"] = y_test
test_data["split"] = "test"

processed_data = pd.concat([train_data, test_data])

# Save processed dataset
processed_data.to_csv(OUTPUT_FILE, index=False)

print("Preprocessing completed successfully!")
print("Processed dataset saved to:", OUTPUT_FILE)