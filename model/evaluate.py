import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv("data/Telecom_data.csv")

# Drop customerID
data = data.drop("customerID", axis=1)

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data = data.dropna()

# Encode target variable
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# One-hot encode categorical features
X = pd.get_dummies(data.drop("Churn", axis=1))
y = data["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load the saved pipeline (scaler + SVM)
model = joblib.load("model/telecom_svm_model.pkl")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")
