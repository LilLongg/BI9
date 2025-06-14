import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


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
    mo.md(r"""### Competitor Dataset""")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Read the dataset""")
    return


@app.cell
def _(DATA_DIR, pl):
    raw_competitor_df = pl.read_csv(
        DATA_DIR / "Competitor database_xlnm#_FilterDatabase.csv", separator=";"
    ).unique()
    raw_competitor_df
    return (raw_competitor_df,)


@app.cell
def _(mo):
    mo.md(r"""#### Check for missing values""")
    return


@app.cell
def _(raw_competitor_df):
    raw_competitor_df.null_count()
    return


@app.cell
def _(mo):
    mo.md(r"""#### Save the result""")
    return


@app.cell
def _(OUTPUT_DIR, raw_competitor_df):
    raw_competitor_df.write_csv(OUTPUT_DIR / "competitor.csv")
    return


@app.cell
def _(mo):
    mo.md(r"""### Customer Dataset""")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Read the dataset""")
    return


@app.cell
def _(DATA_DIR, pl):
    raw_customer_df = (
        pl.read_csv(DATA_DIR / "SA#var.csv", separator=";")
        .unique()
        .drop(["Col", "MPI_Mean_Use", "MPI#2"])
    )
    raw_customer_df
    return (raw_customer_df,)


@app.cell
def _(mo):
    mo.md(r"""#### Check if the ID is unique""")
    return


@app.cell
def _(raw_customer_df):
    raw_customer_df["ID"].unique().len(), raw_customer_df.height
    return


@app.cell
def _(mo):
    mo.md(
        r"""The ID is unique, indicating that each ID is applied for one group in a city in a year."""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""#### Impute missing values""")
    return


@app.cell
def _(raw_customer_df):
    raw_customer_df.null_count()
    return


@app.cell
def _(mo, raw_customer_df):
    mo.md(
        rf"""For attributes `Age` and `Group_size`, since the missing values size is so small ({100 * raw_customer_df.null_count()[0, "Age"] / raw_customer_df.height:.3f}% and {100 * raw_customer_df.null_count()[0, "Group_size"] / raw_customer_df.height:.3f}%), we can impute using `mean`."""
    )
    return


@app.cell
def _(pl, raw_customer_df):
    customer_df = raw_customer_df.with_columns(
        pl.col(["Age", "Group_size"]).fill_null(strategy="mean")
    )
    customer_df["Age", "Group_size"].null_count()
    return (customer_df,)


@app.cell
def _(mo):
    mo.md(r"""After imputing `Age`, we will impute the `Age group`:""")
    return


@app.cell
def _(customer_df):
    customer_df["Age#group", "Age#Group#2"].unique().sort("*")
    return


@app.cell
def _():
    def map_age_group(age: int) -> str:
        if age < 20:
            return "16 - 19"
        if age < 30:
            return "20 - 29"
        if age < 40:
            return "30 - 39"
        return "40 - 60"


    def map_age_group_2(age: int) -> str:
        if age < 20:
            return "16 - 19 y.o."
        if age < 25:
            return "20 - 24 y.o."
        if age < 30:
            return "25 - 29 y.o."
        if age < 35:
            return "30 - 34 y.o."
        if age < 40:
            return "35 - 39 y.o."
        if age < 45:
            return "40 - 44 y.o."
        return "45+ y.o."
    return map_age_group, map_age_group_2


@app.cell
def _(customer_df, map_age_group, map_age_group_2, pl):
    AGE_GROUP_MAPS = {"Age#group": map_age_group, "Age#Group#2": map_age_group_2}

    age_imputed_df = customer_df.with_columns(
        [
            pl.when(pl.col(key).is_null())
            .then(pl.col("Age").map_elements(func, return_dtype=str))
            .otherwise(pl.col(key))
            .alias(key)
            for key, func in AGE_GROUP_MAPS.items()
        ]
    )
    age_imputed_df[*AGE_GROUP_MAPS.keys()].null_count()
    return (age_imputed_df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    For `MPI`, we can take the mean from the same group.

    The group includes:

    - `City`
    - `Gender`
    - `Age group`
    - `Occupation group`

    `Year` is not included since there is no information of the year 2017.
    """
    )
    return


@app.cell
def _(age_imputed_df, pl):
    MPI_IMPUTE_GROUPS = [
        "City",
        "Gender",
        "Age#Group#2",
        "Occupation#group",
    ]

    customer_mpi_non_null = (
        age_imputed_df.filter(~pl.col("MPI#Mean").is_null())
        .group_by(MPI_IMPUTE_GROUPS)
        .agg(
            [
                pl.col("MPI#Mean").mean().cast(int),
                pl.col("City").len().alias("Count"),
            ]
        )
    )
    customer_mpi_non_null
    return MPI_IMPUTE_GROUPS, customer_mpi_non_null


@app.cell
def _(mo):
    mo.md(r"""We use these groups to impute.""")
    return


@app.cell
def _(MPI_IMPUTE_GROUPS, age_imputed_df, customer_mpi_non_null, pl):
    mpi_imputed_df = (
        age_imputed_df.join(
            customer_mpi_non_null, how="left", on=MPI_IMPUTE_GROUPS
        )
        .with_columns(
            pl.when(pl.col("MPI#Mean").is_null())
            .then(pl.col("MPI#Mean_right"))
            .otherwise(pl.col("MPI#Mean"))
            .alias("MPI#Mean")
        )
        .drop(["MPI#Mean_right", "Count"])
    )
    mpi_imputed_df["MPI#Mean"].null_count()
    return (mpi_imputed_df,)


@app.cell
def _(mo, mpi_imputed_df):
    mo.md(
        rf"""There are still missing values, due to some outliers that do not belong to any group. Since the proportion is small ({100 * mpi_imputed_df.null_count()[0, "MPI#Mean"] / mpi_imputed_df.height:.3f}%), we can use mean to impute the rest."""
    )
    return


@app.cell
def _(mpi_imputed_df, pl):
    imputed_customer_df = mpi_imputed_df.with_columns(
        pl.col("MPI#Mean").fill_null(strategy="mean")
    )
    imputed_customer_df["MPI#Mean"].null_count()
    return (imputed_customer_df,)


@app.cell
def _(mo):
    mo.md(r"""With that, we can also impute `MPI Group`""")
    return


@app.cell
def _(imputed_customer_df):
    (
        imputed_customer_df["MPI"].unique().sort(),
        imputed_customer_df["MPI#detail"].unique().sort(),
    )
    return


@app.cell
def _():
    def map_mpi_group(mpi: int) -> str:
        if mpi < 4500:
            return "Under VND 4.5m"
        if mpi < 9000:
            return "VND 4.5m - VND 8.9m"
        if mpi < 15000:
            return "VND 9m - VND 14.9m"
        if mpi < 25000:
            return "VND 15m - VND 24.9m"
        return "VND 25m+"


    def map_mpi_detail(mpi: int) -> str:
        if mpi < 3000:
            return "Under 3 millions VND"
        if mpi < 4500:
            return "From 3 millions to 4.49 millions VND"
        if mpi < 6500:
            return "From 4 millions to 6.49 millions VND"
        if mpi < 7500:
            return "From 6.5 millions to 7.49 millions VND"
        if mpi < 9000:
            return "From 7.5 millions to 8.99 millions VND"
        if mpi < 12000:
            return "From 9 millions to 11.99 millions VND"
        if mpi < 15000:
            return "From 12 millions to 14.99 millions VND"
        if mpi < 20000:
            return "From 15 millions to 19.99 millions VND"
        if mpi < 25000:
            return "From 20 millions to 24.99 millions VND"
        if mpi < 30000:
            return "From 25 millions to 29.99 millions VND"
        if mpi < 45000:
            return "From 30 millions to 44.99 millions VND"
        if mpi < 75000:
            return "From 45 millions to 74.99 millions VND"
        return "From 75 million to VND 149.99 million VND"
    return map_mpi_detail, map_mpi_group


@app.cell
def _(imputed_customer_df, map_mpi_detail, map_mpi_group, pl):
    MPI_GROUP_MAPS = {"MPI": map_mpi_group, "MPI#detail": map_mpi_detail}


    mpi_group_imputed_df = imputed_customer_df.with_columns(
        [
            pl.when(pl.col(key).is_null())
            .then(pl.col("Age").map_elements(func, return_dtype=str))
            .otherwise(pl.col(key))
            .alias(key)
            for key, func in MPI_GROUP_MAPS.items()
        ]
    )
    mpi_group_imputed_df[*MPI_GROUP_MAPS.keys()].null_count()
    return (mpi_group_imputed_df,)


@app.cell
def _(mo):
    mo.md(r"""Finally, we impute `BUMO_previous`.""")
    return


@app.cell
def _(mpi_group_imputed_df):
    mpi_group_imputed_df["BUMO_Previous"].unique().sort()
    return


@app.cell
def _(mo):
    mo.md(
        r"""There is a value of `Don't have nay brands` which we can use to impute."""
    )
    return


@app.cell
def _(mpi_group_imputed_df, pl):
    customer_cleaned_df = mpi_group_imputed_df.fill_null(
        pl.lit("Don't have any brands")
    )
    customer_cleaned_df
    return (customer_cleaned_df,)


@app.cell
def _(mo):
    mo.md(r"""#### Save the output""")
    return


@app.cell
def _(OUTPUT_DIR, customer_cleaned_df):
    customer_cleaned_df.sort("*").write_csv(OUTPUT_DIR / "customer.csv")
    return


@app.cell
def _(mo):
    mo.md(r"""### Companion Dataset""")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Read the dataset""")
    return


@app.cell
def _(DATA_DIR, pl):
    raw_companion_df = (
        pl.read_csv(
            DATA_DIR / "Companion.csv", separator=";", schema_overrides={"ID": str}
        )
        .drop(["City", "Year"])
        .unique()
    )
    raw_companion_df["ID"]
    return (raw_companion_df,)


@app.cell
def _(mo):
    mo.md(r"""#### Check for missing values""")
    return


@app.cell
def _(raw_companion_df):
    raw_companion_df.null_count()
    return


@app.cell
def _(mo):
    mo.md(r"""#### Save the result""")
    return


@app.cell
def _(OUTPUT_DIR, raw_companion_df):
    raw_companion_df.sort("*").write_csv(OUTPUT_DIR / "companion.csv")
    return


@app.cell
def _(mo):
    mo.md(r"""### Brandhealth Dataset""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### Read the dataset

    - We drop `City` and `Year` since it is already defined on the customer dataset.
    """
    )
    return


@app.cell
def _(DATA_DIR, pl):
    raw_brand_health_df = (
        pl.read_csv(
            DATA_DIR / "Brandhealth.csv",
            separator=";",
            schema_overrides={
                "PPA": float,
                "Fre#visit": int,
                "NPS#P3M": int,
                "Spending": int,
                "Spending_use": int,
            },
        )
        .drop(["City", "Year"])
        .unique()
    )
    raw_brand_health_df
    return (raw_brand_health_df,)


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
        r"""There are no results. Indicating that they are the same. Therefore we will remove $Spending_use$."""
    )
    return


@app.cell
def _(raw_brand_health_df):
    brand_health_df = raw_brand_health_df.drop("Spending_use")
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
        rf"""For `Comprehension`, since there is no way to deduce its value, and it takes up a large proportion of the dataset ({100 * brand_health_df["Comprehension"].null_count() / brand_health_df.height:.2f}%), we can consider dropping the column."""
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
    brand_health_df_6.filter(
        (pl.col("Awareness") == False) & (pl.col("Trial") == True)
    )
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
    brand_health_df_8.filter(
        (pl.col("Fre#visit") != 0) ^ (pl.col("Spending") != 0)
    )
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
    mo.md(r"""Let's check all of the `Brand`s""")
    return


@app.cell
def _(brand_health_df_9):
    brand_health_df_9["Brand"].unique().sort()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    We can see that there is a value of `Street`, which does not appear on the customer dataset.

    We can modify its value to `Street / Half street coffee (including carts)`.
    """
    )
    return


@app.cell
def _(brand_health_df_9, pl):
    cleaned_brand_health_df = brand_health_df_9.with_columns(pl.col("Brand").replace("Street", "Street / Half street coffee (including carts)"))
    cleaned_brand_health_df["Brand"].unique().sort()
    return (cleaned_brand_health_df,)


@app.cell
def _(mo):
    mo.md(r"""#### Save the result""")
    return


@app.cell
def _(OUTPUT_DIR, cleaned_brand_health_df):
    cleaned_brand_health_df.sort("*").write_csv(OUTPUT_DIR / "Brand Health.csv")
    return


@app.cell
def _(mo):
    mo.md(r"""### Day of Week Dataset""")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Read the dataset""")
    return


@app.cell
def _(DATA_DIR, pl):
    raw_dow_df = pl.read_csv(DATA_DIR / "Dayofweek.csv", separator=";").unique()
    raw_dow_df
    return


if __name__ == "__main__":
    app.run()
