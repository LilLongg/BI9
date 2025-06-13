import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Initializing""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Importing packages""")
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    from pathlib import Path
    return Path, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Setting up global variables""")
    return


@app.cell
def _(Path):
    DATA_DIR = Path("../Dataset Round 01/Data")
    OUTPUT_DIR = Path("../Dataset Round 01/Cleaned")

    SEGMENTATION_DATASET = "2017Segmentation3685case.csv"
    BRAND_IMAGE_DATASET = "Brand_Image.csv"
    BRAND_HEALTH_DATASET = "Brandhealth.csv"
    COMPANION_DATASET = "Companion.csv"
    COMPETITOR_DATASET = "Competitor database_xlnm#_FilterDatabase.csv"
    DAYOFWEEK_DATASET = "Dayofweek.csv"
    DAYPART_DATASET = "Daypart.csv"
    NEEDSTATE_DATASET = "NeedstateDayDaypart.csv"
    SA_DATASET = "SA#var.csv"
    return DATA_DIR, OUTPUT_DIR, SEGMENTATION_DATASET


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Cleaning data""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Segmentation dataset""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Read the dataset""")
    return


@app.cell
def _(DATA_DIR, SEGMENTATION_DATASET, pl):
    df = pl.read_csv(DATA_DIR / SEGMENTATION_DATASET, separator=";")
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Check for abnormality""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Check for null values""")
    return


@app.cell
def _(df):
    df.null_count()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Check if $\cfrac{\text{Spending}}{\text{Visit}} = \text{PPA}$.""")
    return


@app.cell
def _(df, pl):
    df.filter(pl.col("Spending") / pl.col("Visit") != pl.col("PPA"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Check if the segmentation is in correct format""")
    return


@app.cell
def _(df):
    df.select("Segmentation").unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Split the segmentation column into a new dataset with four columns:

    - Segmentation ID
    - Segmentation name
    - Average spending range (min, and max).
    """
    )
    return


@app.cell
def _(df, np, pl):
    segmentation_df = pl.DataFrame({'Segmentation ID': [1, 2, 3, 4], 'Segmentation Name': df.select('Segmentation').unique().sort('Segmentation').to_series(), 'Average Min': [-np.inf, 25, 60, 100], 'Average Max': [25, 60, 100, np.inf]}, strict=False)
    df_1 = df.join(segmentation_df, left_on='Segmentation', right_on='Segmentation Name')
    df_1
    return df_1, segmentation_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Check if the PPA is in correct range.""")
    return


@app.cell
def _(df_1, pl):
    df_1.filter([pl.col('PPA') >= pl.col('Average Max'), pl.col('PPA') < pl.col('Average Min')])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can conclude that no abnormality is found in the dataset.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Save the output

    - We save both the segmentation dataset and filter out unnecessary columns in `df`, which include:
      - `Segmentation` (use the ID instead).
      - `Average Min and Max` (since they are from the segmentation dataset).
      - `Speding`, since we can derive it using the formula $\text{Spending} = \text{PPA}\cdot\text{Visit}$.
    """
    )
    return


@app.cell
def _(OUTPUT_DIR, SEGMENTATION_DATASET, df_1, segmentation_df):
    segmentation_df.write_csv(OUTPUT_DIR / 'Segmentation Info.csv')
    df_1.select(['ID', 'Visit', 'PPA', 'Segmentation ID', 'Brand']).write_csv(OUTPUT_DIR / SEGMENTATION_DATASET)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
