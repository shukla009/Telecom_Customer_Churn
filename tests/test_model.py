import unittest
import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pandas as pd

class TestSVMModelTraining(unittest.TestCase):
    def test_model_pipeline(self):
        # Load the saved pipeline
        model = joblib.load("model/telecom_svm_model.pkl")
        
        # Check that it is a Pipeline
        self.assertIsInstance(model, Pipeline)
        
        # Check that the final step is an SVC
        self.assertIsInstance(model.named_steps["svm"], SVC)

    def test_feature_count(self):
        # Load the dataset to confirm number of features
        data = pd.read_csv("data/Telecom_data.csv")
        data = data.drop("customerID", axis=1)
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
        data = data.dropna()
        data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
        X = pd.get_dummies(data.drop("Churn", axis=1))
        
        # Load the pipeline
        model = joblib.load("model/telecom_svm_model.pkl")
        
        # Check that input feature size matches
        self.assertEqual(model.named_steps["scaler"].n_features_in_, X.shape[1])

if __name__ == "__main__":
    unittest.main()
