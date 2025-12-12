import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from utils.model_fit import fit_knn_regressor, main

@pytest.fixture
def dummy_train_data():
    return pd.DataFrame({
        'Feature1': [1.0, 2.0, 3.0, 4.0],
        'Feature2': [5.0, 6.0, 7.0, 8.0],
        'Rings': [10, 20, 30, 40]
    })

@pytest.fixture
def runner():
    return CliRunner()

def test_fit_knn_regressor_types_and_params(dummy_train_data):
    X_train = dummy_train_data.drop("Rings", axis=1)
    y_train = dummy_train_data["Rings"]
    
    knn, scaler = fit_knn_regressor(X_train, y_train, n_neighbors=7)
    
    assert isinstance(knn, KNeighborsRegressor)
    assert isinstance(scaler, StandardScaler)
    assert knn.n_neighbors == 7
    assert knn.n_features_in_ == 2
    assert scaler.n_features_in_ == 2

def test_main_cli_success(runner, tmpdir, dummy_train_data):
    train_path = tmpdir.join("train_data.csv")
    dummy_train_data.to_csv(train_path, index=False)
    model_output = str(tmpdir.join("model.pkl"))
    scaler_output = str(tmpdir.join("scaler.pkl"))
    
    mock_knn = MagicMock(spec=KNeighborsRegressor)
    mock_scaler = MagicMock(spec=StandardScaler)
    
    with patch("utils.model_fit.fit_knn_regressor", return_value=(mock_knn, mock_scaler)) as mock_fit:
        with patch("utils.model_fit.pickle.dump") as mock_dump:
        
            result = runner.invoke(
                main,
                [
                    "--train_path", train_path, 
                    "--model_output", model_output,
                    "--scaler_output", scaler_output,
                    "--n_neighbors", "10" 
                ]
            )
            
            assert result.exit_code == 0
            args, kwargs = mock_fit.call_args
            assert args[2] == 10
            assert mock_dump.call_count == 2
            
            import os
            assert os.path.exists(model_output)
            assert os.path.exists(scaler_output)
            assert "Model fitting completed successfully." in result.output

def test_main_cli_missing_input(runner):
    result = runner.invoke(
        main, 
        [
            "--model_output", "dummy_model.pkl",
            "--scaler_output", "dummy_scaler.pkl"
        ]
    )
    
    assert result.exit_code != 0
    assert "Error: Missing option '--train_path'" in result.output
    assert "Model fitting completed successfully." not in result.output
    print("\n✅ CLI Missing Input Test Passed: Correctly failed on missing argument.")