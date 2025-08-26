import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
data = pd.read_csv("data/Telecom_data.csv")

# Drop customerID (not useful)
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

# Build pipeline (scaler + SVM)
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1, gamma="scale", random_state=42))
])

# Train pipeline
svm_pipeline.fit(X_train, y_train)

# Save pipeline as ONE file
joblib.dump(svm_pipeline, "model/telecom_svm_model.pkl")
