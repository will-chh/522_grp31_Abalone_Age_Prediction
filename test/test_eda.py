import pytest
import altair as alt
import pandas as pd
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from utils.data_eda import scatter_matrix, NEW_COLUMN_NAMES, main


@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def dummy_input_data():
    return pd.DataFrame({
        "Length": [0.1, 0.2, 0.3],
        "Diameter": [0.3, 0.4, 0.5],
        "Height": [0.5, 0.6, 0.7],
        "Whole_weight": [0.7, 0.8, 0.9],
        "Shucked_weight": [0.9, 1.0, 1.1],
        "Viscera_weight": [1.1, 1.2, 1.3],
        "Shell_weight": [1.3, 1.4, 1.5],
        "Rings": [8, 9, 10],
        "Sex": ["M", "F", "I"]
    })

# --- Tests ---
def test_scatter_matrix(dummy_input_data):
    chart = scatter_matrix(dummy_input_data)
    assert isinstance(chart, alt.RepeatChart)
    repeat_spec = chart.to_dict()["repeat"]
    assert repeat_spec["row"] == NEW_COLUMN_NAMES
    assert repeat_spec["column"] == NEW_COLUMN_NAMES
    encoding = chart.to_dict()["spec"]["encoding"]
    assert encoding["color"]["field"] == "Sex"
    assert encoding["color"]["type"] == "nominal"
    assert encoding["color"]["title"] == "Abalone Sex" 
    print("\n✅ Scatter Matrix Unit Test Passed: Chart structure verified.")

def test_main_cli_success(runner, tmpdir, dummy_input_data):
    input_path = tmpdir.join("validated_input.csv")
    dummy_input_data.to_csv(input_path, index=False)
    output_path = tmpdir.join("scatter_matrix.json")
    mock_chart = MagicMock(spec=alt.Chart)

    with patch("utils.data_eda.scatter_matrix", return_value=mock_chart) as mock_scatter_matrix:
        result = runner.invoke(
            main,
            ["--input_path", str(input_path), "--output_path", str(output_path)]
        )

        assert result.exit_code == 0
        mock_scatter_matrix.assert_called_once()
        assert isinstance(mock_scatter_matrix.call_args[0][0], pd.DataFrame)
        mock_chart.save.assert_called_once_with(str(output_path))
        assert f"Scatter matrix saved to {output_path}" in result.output
        
        print("\n✅ CLI Success Test Passed: Logic called and chart.save() verified.")

def test_main_cli_missing_input(runner):
    result = runner.invoke(main, ["--output_path", "output.json"])
    assert result.exit_code != 0
    assert "Error: Missing option '--input_path'" in result.output
    print("\n✅ CLI Missing Input Test Passed: Correctly failed on missing argument.")