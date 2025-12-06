import altair as alt
import pandas as pd

NEW_COLUMN_NAMES = [
    "Length", "Diameter", "Height",
    "Whole_weight", "Shucked_weight",
    "Viscera_weight", "Shell_weight", "Rings"
]

def scatter_matrix(df: pd.DataFrame) -> alt.Chart:
    chart = (
        alt.Chart(df, width=150, height=100)
        .mark_point()
        .encode(
            alt.X(alt.repeat("row"), type="quantitative"),
            alt.Y(alt.repeat("column"), type="quantitative"),
            color=alt.Color("Sex:N", title="Abalone Sex"),
        )
        .repeat(column=NEW_COLUMN_NAMES, row=NEW_COLUMN_NAMES)
        .properties(title="Scatterplot matrix of abalone physical features and rings")
    )
    return chart