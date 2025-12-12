import sys
import os
import pandas as pd
from unittest.mock import patch

sys.path.append("../")
from utils.model_preprocess import preprocess_and_split

def test_preprocess_and_split_basic():
    # dummy df -- deepcheck requires min 10 values
    df = pd.DataFrame({
        "Length": [0.1,0.2,0.15,0.18,0.22,0.25,0.3,0.35,0.4,0.45],
        "Diameter": [0.3,0.4,0.35,0.38,0.32,0.36,0.42,0.37,0.41,0.44],
        "Height": [0.5,0.6,0.55,0.58,0.52,0.57,0.62,0.59,0.63,0.65],
        "Whole_weight": [0.7,0.8,0.75,0.78,0.72,0.77,0.82,0.79,0.83,0.85],
        "Shucked_weight": [0.9,1.0,0.95,0.98,0.92,0.97,1.02,0.99,1.03,1.05],
        "Viscera_weight": [1.1,1.2,1.15,1.18,1.12,1.17,1.22,1.19,1.23,1.25],
        "Shell_weight": [1.3,1.4,1.35,1.38,1.32,1.37,1.42,1.39,1.43,1.45],
        "Rings": [8,9,7,10,9,8,11,7,12,9],
        "Sex": ["M","F","I","M","F","I","M","F","I","M"]
    })

    with patch("utils.model_preprocess.Dataset"), \
         patch("utils.model_preprocess.FeatureLabelCorrelation") as mock_feat_lab, \
         patch("utils.model_preprocess.FeatureFeatureCorrelation") as mock_feat_feat, \
         patch("utils.model_preprocess.LabelDrift") as mock_label_drift:

        for mock_check in [mock_feat_lab, mock_feat_feat, mock_label_drift]:
            mock_instance = mock_check.return_value
            mock_instance.run.return_value.passed_conditions.return_value = True

        X_train, X_test, y_train, y_test = preprocess_and_split(
            df, test_size=0.5, random_state=42
        )

        # Check train/test shapes
        assert X_train.shape[0] == 5
        assert X_test.shape[0] == 5
        assert y_train.shape[0] == 5
        assert y_test.shape[0] == 5

        # Check that train/test have same columns
        assert set(X_train.columns) == set(X_test.columns)

        expected_sex_dummies = [c for c in X_train.columns if c.startswith("Sex_")]

        assert set(expected_sex_dummies) == {"Sex_I", "Sex_M"}

        assert all(y_train.isin(df["Rings"]))
        assert all(y_test.isin(df["Rings"]))

    print("TEST: model preprocess PASSED")

# Run test directly
if __name__ == "__main__":
    test_preprocess_and_split_basic()
