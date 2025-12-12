import os
import pandas as pd
import pytest
from unittest.mock import patch
from click.testing import CliRunner
from utils.model_preprocess import preprocess_and_split, main

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def df_fixture():
    return pd.DataFrame({
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

@pytest.fixture
def mock_split_output(df_fixture):
    df = df_fixture.drop(columns=['Sex', 'Rings'])
    df['Sex_I'] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    df['Sex_M'] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    X_train = df.iloc[:5].reset_index(drop=True)
    X_test = df.iloc[5:].reset_index(drop=True)
    y_train = df_fixture['Rings'].iloc[:5].reset_index(drop=True)
    y_test = df_fixture['Rings'].iloc[5:].reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test

def test_preprocess_and_split_basic(df_fixture):
    df = df_fixture
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

def test_main_cli_success(runner, tmpdir, df_fixture, mock_split_output):
    input_path = tmpdir.join("validated_input.csv")
    df_fixture.to_csv(input_path, index=False)
    train_output = str(tmpdir.join("train.csv"))
    test_output = str(tmpdir.join("test.csv"))

    with patch("utils.model_preprocess.preprocess_and_split", return_value=mock_split_output) as mock_preprocess:
        
        result = runner.invoke(
            main,
            [
                "--input_path", str(input_path), 
                "--train_output", train_output,
                "--test_output", test_output,
                "--test_size", "0.3", 
                "--random_state", "100"
            ]
        )
        
        assert result.exit_code == 0
        
        mock_preprocess.assert_called_once()
        args, kwargs = mock_preprocess.call_args
        assert args[1] == 0.3
        assert args[2] == 100
        
        assert os.path.exists(train_output)
        assert os.path.exists(test_output)
        loaded_train = pd.read_csv(train_output)
        assert loaded_train.shape[0] == 5
        assert 'Rings' in loaded_train.columns

        expected_columns = [
            'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 
            'Viscera_weight', 'Shell_weight', 'Sex_I', 'Sex_M', 'Rings'
        ]

        assert set(loaded_train.columns) == set(expected_columns)
        assert 'Length' in loaded_train.columns
        
        print("\n✅ CLI Success Test Passed: Files created and logic called correctly.")

def test_main_cli_missing_input(runner):
    result = runner.invoke(
        main, 
        [
            "--train_output", "dummy_train.csv",
            "--test_output", "dummy_test.csv"
        ]
    )
    assert result.exit_code != 0
    assert "Error: Missing option '--input_path'" in result.output
    print("\n✅ CLI Missing Input Test Passed: Correctly failed on missing argument.")
