import pytest
import numpy as np
import pandas as pd
import pickle
from unittest.mock import patch, MagicMock
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from click.testing import CliRunner
from utils.model_eval import evaluate_knn, main

@pytest.fixture
def dummy_eval_data():
    X = pd.DataFrame({'f1': [1, 2, 3, 4], 'f2': [5, 6, 7, 8]})
    y = pd.Series([10, 20, 30, 40])
    return X, y

@pytest.fixture
def mock_model_and_scaler():
    mock_knn = MagicMock(spec=KNeighborsRegressor)
    mock_scaler = MagicMock(spec=StandardScaler)
    return mock_knn, mock_scaler

@pytest.fixture
def runner():
    return CliRunner()

def test_evaluate_knn_calculation_and_plotting(tmpdir, dummy_eval_data, mock_model_and_scaler):
    X, y = dummy_eval_data
    X_train, X_test = X.iloc[:2], X.iloc[2:]
    y_train, y_test = y.iloc[:2], y.iloc[2:]
    
    mock_knn, mock_scaler = mock_model_and_scaler
    plot_path = tmpdir.join("test_plot.png")
    mock_scaler.transform.return_value = np.array([[0, 0], [1, 1]]) 
    
    mock_knn.predict.side_effect = [
        np.array([11, 21]),
        np.array([32, 42]),
    ]

    with patch("matplotlib.pyplot.savefig") as mock_save:
        train_rmse, test_rmse = evaluate_knn(
            mock_knn, mock_scaler, X_train, y_train, X_test, y_test, plot_path
        )
        
        assert np.isclose(train_rmse, 1.0)
        assert np.isclose(test_rmse, 2.0)
        mock_save.assert_called_once_with(plot_path, bbox_inches="tight")

def test_main_cli_success(runner, tmpdir, dummy_eval_data):
    X, y = dummy_eval_data
    train_df = pd.concat([X.iloc[:2], y.iloc[:2].rename('Rings')], axis=1)
    test_df = pd.concat([X.iloc[2:], y.iloc[2:].rename('Rings')], axis=1)

    train_path = tmpdir.join("train.csv")
    test_path = tmpdir.join("test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    model_path = tmpdir.join("model.pkl")
    scaler_path = tmpdir.join("scaler.pkl")
    with open(model_path, "wb") as f:
        pickle.dump("DUMMY_MODEL", f)
    with open(scaler_path, "wb") as f:
        pickle.dump("DUMMY_SCALER", f)

    plot_output = tmpdir.join("eval_plot.png")
    
    with patch("utils.model_eval.evaluate_knn") as mock_eval:
        result = runner.invoke(
            main,
            [
                "--train_path", str(train_path), 
                "--test_path", str(test_path),
                "--model_path", str(model_path),
                "--scaler_path", str(scaler_path),
                "--plot_output", str(plot_output),
            ]
        )

        assert result.exit_code == 0
        mock_eval.assert_called_once()
        args, kwargs = mock_eval.call_args
        assert kwargs['plot_output'] == str(plot_output)
        assert isinstance(kwargs['X_train'], pd.DataFrame)
        assert kwargs['X_train'].shape[0] == 2
        
        print("\n✅ CLI Success Test Passed: File I/O and logic call verified.")


def test_main_cli_missing_path(runner):
    result = runner.invoke(main, ["--train_path", "dummy.csv"])
    assert result.exit_code != 0
    assert "Error: Missing option" in result.output
    print("\n✅ CLI Missing Path Test Passed: Correctly failed on missing argument.")