"""
Functions to visualization
"""
import pandas as pd
import plotly.express as px


def highlighting_dots(
    df: pd.DataFrame,
    x: str,
    y: str,
    indexes_list: list,
    marker_size=10,
    marker_color="red",
):
    """
    df: DataFrame to get the plot
    x: Name of the column to be plotted on X axis
    y: Name of the column to be plotted on Y axis
    indexes_list: List cointaning the index of the points the
    we'd like to emphasize
    """
    grafico = px.scatter(df, x=x, y=y)
    grafico.add_traces(
        px.scatter(df.iloc[indexes_list], x=x, y=y)
        .update_traces(marker_size=marker_size, marker_color=marker_color)
        .data
    )
    grafico.show()
