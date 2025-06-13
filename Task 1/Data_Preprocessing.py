import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _(mo):
    mo.md(r"""## Initialize""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Import necessary packages""")
    return


@app.cell
def _():
    import numpy as np
    import marimo as mo
    import polars as pl
    from pathlib import Path
    import re

    return Path, mo, pl


@app.cell
def _(mo):
    mo.md(r"""### Setup global variables""")
    return


@app.cell
def _(Path):
    DATA_DIR = Path("../Dataset Round 01/Data")
    OUTPUT_DIR = Path("../Dataset Round 01/Cleaned")
    return DATA_DIR, OUTPUT_DIR


@app.cell
def _(mo):
    mo.md(r"""## Preprocessing""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Brandhealth Dataset""")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Read the dataset""")
    return


@app.cell
def _(DATA_DIR, pl):
    raw_brand_health_df = pl.read_csv(
        DATA_DIR / "Brandhealth.csv",
        separator=";",
        schema_overrides={
            "PPA": float,
            "Fre#visit": int,
            "NPS#P3M": int,
            "Spending": int,
            "Spending_use": int,
        },
    ).unique()
    raw_brand_health_df
    return (raw_brand_health_df,)


@app.cell
def _(mo):
    mo.md(r"""#### Check if each ID applies for one year and one city""")
    return


@app.cell
def _(raw_brand_health_df):
    (
        raw_brand_health_df.unique("ID").height,
        raw_brand_health_df.unique(["ID", "Year", "City"]).height,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""The number of unique rows with `ID`, and the tuple (`ID`, `Year`, `City`) is the same, suggesting that `Year` and `City` can be derived from a unique `ID`."""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""#### Remove redundant columns""")
    return


@app.cell
def _(mo):
    mo.md(r"""Check if `Spending` is the same as `Spending_use`""")
    return


@app.cell
def _(pl, raw_brand_health_df):
    raw_brand_health_df.filter(pl.col("Spending") != pl.col("Spending_use"))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    There are no results. Indicating that they are the same. Therefore we will remove $Spending_use$.

    We will also remove `City` and `Year` for obvious reasons.
    """
    )
    return


@app.cell
def _(raw_brand_health_df):
    brand_health_df = raw_brand_health_df.drop(["City", "Year", "Spending_use"])
    return (brand_health_df,)


@app.cell
def _(mo):
    mo.md(r"""#### Impute missing values""")
    return


@app.cell
def _(brand_health_df):
    brand_health_df.null_count()
    return


@app.cell
def _(mo):
    mo.md(r"""There are a lot of missing values. Let's try to impute them.""")
    return


@app.cell
def _(brand_health_df, mo):
    mo.md(
        rf"""For `Comprehension`, since there is no way to deduce its value, and it takes up a large proportion of the dataset ({100 * brand_health_df["Comprehension"].null_count() / brand_health_df.height:.2f}%), we can consider droping the column."""
    )
    return


@app.cell
def _(brand_health_df):
    brand_health_df_1 = brand_health_df.drop("Comprehension")
    return (brand_health_df_1,)


@app.cell
def _(mo):
    mo.md(
        r"""For statistical attributes (`Fre`, `PPA`, `NPS`), missing values could indicate that the value is 0."""
    )
    return


@app.cell
def _(brand_health_df_1, pl):
    NUMERICAL_ATTRIBUTES = ["Fre#visit", "PPA", "NPS#P3M", "Spending"]

    brand_health_df_2 = brand_health_df_1.with_columns(
        pl.col(NUMERICAL_ATTRIBUTES).fill_null(0)
    )
    brand_health_df_2.select(NUMERICAL_ATTRIBUTES).null_count()
    return NUMERICAL_ATTRIBUTES, brand_health_df_2


@app.cell
def _(mo):
    mo.md(
        r"""
    For categorical attributes (`Segmentation` and `NPS`), the value can be imputed using the corresponding attributes.

    Let's first check for all of the possible attributes:
    """
    )
    return


@app.cell
def _(brand_health_df_2):
    (
        brand_health_df_2["Segmentation"].unique().sort(),
        brand_health_df_2["NPS#P3M#Group"].unique().sort(),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""We then create a mapper function for each category:""")
    return


@app.cell
def _():
    def segmenatation_map(ppa: int) -> str:
        if ppa < 25:
            return "Seg.01 - Mass (<VND 25K)"
        if ppa < 60:
            return "Seg.02 - Mass Asp (VND 25K - VND 59K)"
        if ppa < 100:
            return "Seg.03 - Premium (VND 60K - VND 99K)"
        return "Seg.04 - Super Premium (VND 100K+)"

    def nps_map(nps: int) -> str:
        if nps < 7:
            return "Detractor"
        if nps < 9:
            return "Passive"
        return "Promoter"

    return nps_map, segmenatation_map


@app.cell
def _(mo):
    mo.md(r"""We use the functions to impute the categorical attributes""")
    return


@app.cell
def _(NUMERICAL_ATTRIBUTES, brand_health_df_2, nps_map, pl, segmenatation_map):
    CATEGORICAL_MAPPING = [
        ("Segmentation", "PPA", segmenatation_map),
        ("NPS#P3M#Group", "NPS#P3M", nps_map),
    ]

    brand_health_df_3 = brand_health_df_2.with_columns(
        [
            pl.when(pl.col(category).is_null())
            .then(pl.col(value).map_elements(func, return_dtype=str))
            .otherwise(pl.col(category))
            .alias(category)
            for category, value, func in CATEGORICAL_MAPPING
        ]
    )
    brand_health_df_3.select(NUMERICAL_ATTRIBUTES).null_count()
    return (brand_health_df_3,)


@app.cell
def _(mo):
    mo.md(
        r"""For `Spontaneous`, `Awareness`, `Brand_Likability`, and the boolean columns; their values, if not missing, is one of `Brand`:"""
    )
    return


@app.cell
def _(brand_health_df_3):
    BOOLEAN_COLUMNS = [
        "Spontaneous",
        "Awareness",
        "Trial",
        "P3M",
        "P1M",
        "Brand_Likability",
    ]

    for column in ["Brand"] + BOOLEAN_COLUMNS:
        print(brand_health_df_3[column].drop_nulls().unique().sort())
    return (BOOLEAN_COLUMNS,)


@app.cell
def _(mo):
    mo.md(r"""Therefore, we use that rule to impute the columns.""")
    return


@app.cell
def _(BOOLEAN_COLUMNS, brand_health_df_3, pl):
    brand_health_df_4 = brand_health_df_3.with_columns(
        [
            pl.when(pl.col(attribute) == pl.col("Brand"))
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias(attribute)
            for attribute in BOOLEAN_COLUMNS
        ]
    )
    brand_health_df_4.select(BOOLEAN_COLUMNS).null_count()
    return (brand_health_df_4,)


@app.cell
def _(mo):
    mo.md(r"""For `Weekly` and `Daily`, we also have to check for `Applicable`.""")
    return


@app.cell
def _(brand_health_df_4, pl):
    DAYS_OF_WEEK_COL = ["Weekly", "Daily"]

    brand_health_df_4[DAYS_OF_WEEK_COL].filter(
        (pl.col("Weekly") == "Applicable") | (pl.col("Daily") == "Applicable")
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""Since there is no brand named `Not Applicable`, we can use the same method as above."""
    )
    return


@app.cell
def _(brand_health_df_4, pl):
    brand_health_df_5 = brand_health_df_4.with_columns(
        [
            pl.when(pl.col(attribute) == pl.col("Brand"))
            .then(pl.lit("Applicable"))
            .otherwise(pl.lit("Not Applicable"))
            .alias(attribute)
            for attribute in ["Weekly", "Daily"]
        ]
    )
    brand_health_df_5.select(["Weekly", "Daily"]).null_count()
    return (brand_health_df_5,)


@app.cell
def _(mo):
    mo.md(r"""It seems that we are done. Let's check the result.""")
    return


@app.cell
def _(brand_health_df_5):
    brand_health_df_5.null_count()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### Logical Correction

    We have to make sure that:

    - If $\text{Spontaneous} = \text{True}$, then $\text{Awareness} = \text{True}$.

    - If $\text{Trial} = \text{True}$, then $\text{Awareness} = True$.

    - If $\text{P1M} = \text{True}$, then $\text{P3M} = \text{True}$.

    - If $\text{Daily} = \text{True}$, then $\text{Weekly} = \text{True}$.

    - $\text{Spending} = 0 \Leftrightarrow \text{Fre} = 0$.

    - The formula for $\text{PPA}$ is correct.

    - `Segmentation` and `NPS` is grouped correctly.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""First task:""")
    return


@app.cell
def _(brand_health_df_5, pl):
    brand_health_df_6 = brand_health_df_5.filter(
        (pl.col("Awareness") == True) | (pl.col("Spontaneous") == False)
    )
    brand_health_df_5.filter(
        (pl.col("Awareness") == False) & (pl.col("Spontaneous") == True)
    )
    return (brand_health_df_6,)


@app.cell
def _(mo):
    mo.md(r"""Second task:""")
    return


@app.cell
def _(brand_health_df_6, pl):
    brand_health_df_7 = brand_health_df_6.filter(
        (pl.col("Awareness") == True) | (pl.col("Trial") == False)
    )
    brand_health_df_6.filter((pl.col("Awareness") == False) & (pl.col("Trial") == True))
    return (brand_health_df_7,)


@app.cell
def _(mo):
    mo.md(r"""Third task:""")
    return


@app.cell
def _(brand_health_df_7, pl):
    brand_health_df_8 = brand_health_df_7.filter(
        (pl.col("P3M") == True) | (pl.col("P1M") == False)
    )
    brand_health_df_7.filter((pl.col("P3M") == False) & (pl.col("P1M") == True))
    return (brand_health_df_8,)


@app.cell
def _(mo):
    mo.md(r"""Third task:""")
    return


@app.cell
def _(brand_health_df_8, pl):
    brand_health_df_8.filter((pl.col("P3M") == False) & (pl.col("P1M") == True))
    return


@app.cell
def _(mo):
    mo.md(r"""Fourth task:""")
    return


@app.cell
def _(brand_health_df_8, pl):
    brand_health_df_9 = brand_health_df_8.filter(
        (pl.col("Fre#visit") == 0) ^ (pl.col("Spending") != 0)
    )
    brand_health_df_8.filter((pl.col("Fre#visit") != 0) ^ (pl.col("Spending") != 0))
    return (brand_health_df_9,)


@app.cell
def _(mo):
    mo.md(r"""Fifth task:""")
    return


@app.cell
def _(brand_health_df_9, pl):
    brand_health_df_9.filter(
        pl.col("PPA")
        != pl.when(pl.col("Fre#visit") == 0)
        .then(pl.lit(0))
        .otherwise((pl.col("Spending") / pl.col("Fre#visit")))
        .round(1)
    )
    return


@app.cell
def _(mo):
    mo.md(r"""Sixth task:""")
    return


@app.cell
def _(brand_health_df_9, nps_map, pl, segmenatation_map):
    brand_health_df_9.filter(
        (
            pl.col("Segmentation")
            != pl.col("PPA").map_elements(segmenatation_map, return_dtype=str)
        )
        | (
            pl.col("NPS#P3M#Group")
            != pl.col("NPS#P3M").map_elements(nps_map, return_dtype=str)
        )
    )
    return


@app.cell
def _(mo):
    mo.md(r"""#### Save the result""")
    return


@app.cell
def _(OUTPUT_DIR, brand_health_df_9):
    brand_health_df_9.write_csv(OUTPUT_DIR / "Brand Health.csv")
    return


if __name__ == "__main__":
    app.run()
