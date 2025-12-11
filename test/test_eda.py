import importlib.util
import pathlib
import altair as alt
import pandas as pd


import sys
import os
sys.path.append("../")
from utils.data_eda import scatter_matrix, NEW_COLUMN_NAMES


# --- Tests ---
def test_scatter_matrix():
    # minimal dummy dataframe
    df = pd.DataFrame({
        "Length": [0.1, 0.2],
        "Diameter": [0.3, 0.4],
        "Height": [0.5, 0.6],
        "Whole_weight": [0.7, 0.8],
        "Shucked_weight": [0.9, 1.0],
        "Viscera_weight": [1.1, 1.2],
        "Shell_weight": [1.3, 1.4],
        "Rings": [8, 9],
        "Sex": ["M", "F"]
    })

    chart = scatter_matrix(df)

    # Check chart type
    assert isinstance(chart, alt.RepeatChart)

    # Check repeat structure
    repeat_spec = chart.to_dict()["repeat"]

    assert repeat_spec["row"] == NEW_COLUMN_NAMES
    assert repeat_spec["column"] == NEW_COLUMN_NAMES


    # Check color encoding
    encoding = chart.to_dict()["spec"]["encoding"]
    assert encoding["color"]["field"] == "Sex"
    assert encoding["color"]["type"] == "nominal"
    print("hello")


test_scatter_matrix()