import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from feature_extractor import extract_features


# Load dataset
data = pd.read_csv("dataset.csv")

print("Dataset loaded successfully.")
print(f"Total URLs: {len(data)}")


# Extract numerical features from every URL
X = pd.DataFrame(
    [extract_features(url) for url in data["url"]]
)

y = data["label"]


# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Create Random Forest AI model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train
model.fit(X_train, y_train)


# Test
predictions = model.predict(X_test)


# Evaluation
accuracy = accuracy_score(y_test, predictions)

print("\n===== AI CYBERSHIELD MODEL EVALUATION =====")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Phishing"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


# Feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nMost Important Features:")
print(importance.to_string(index=False))


# Save trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nTrained model saved as model.pkl")