import pytest
import pandas as pd
import sys
import os

from unittest.mock import patch
from pandera.errors import SchemaErrors

sys.path.append('../')
from utils.data_import import load_and_validate_abalone, main


@pytest.fixture
def mock_abalone_df_valid():
    return pd.DataFrame({
        'Sex': ['M', 'F', 'I'],
        'Length': [0.455, 0.530, 0.3],
        'Diameter': [0.365, 0.420, 0.2],
        'Height': [0.095, 0.135, 0.05],
        'Whole_weight': [0.5140, 0.6770, 0.1],
        'Shucked_weight': [0.2245, 0.2565, 0.05],
        'Viscera_weight': [0.1010, 0.1415, 0.03],
        'Shell_weight': [0.150, 0.210, 0.05],
        'Rings': [15, 9, 7]
    })

@pytest.fixture
def mock_abalone_df_invalid():
    return pd.DataFrame({
        'Sex': ['M'],
        'Length': [0.455],
        'Diameter': [0.365],
        'Height': [0.095],
        'Whole_weight': [0.5140],
        'Shucked_weight': [0.2245],
        'Viscera_weight': [0.1010],
        'Shell_weight': [0.150],
        'Rings': [0]
    })

def test_load_and_validate_abalone_success(mock_abalone_df_valid):
    with patch("pandas.read_csv") as mock_read:
        mock_read.return_value = mock_abalone_df_valid
        result = load_and_validate_abalone("dummy_url.csv")
        assert not result.empty
        assert len(result) == 3

def test_load_and_validate_abalone_schema_error(mock_abalone_df_invalid):
    with patch("pandas.read_csv") as mock_read:
        mock_read.return_value = mock_abalone_df_invalid

        with pytest.raises(SchemaErrors) as excinfo:
            load_and_validate_abalone("dummy_url.csv")
        assert "Rings" in str(excinfo.value)
        print("\n✅ Error Test Passed: Invalid data correctly raised SchemaError.")