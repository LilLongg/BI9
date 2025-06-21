import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import polars.selectors as cs
    from prince import FAMD
    import seaborn as sns
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from yellowbrick.cluster import KElbowVisualizer

    plt.style.use("ggplot")
    return FAMD, KElbowVisualizer, KMeans, Path, cs, go, np, pl, plt, sns


@app.cell
def _(Path, np):
    DATA_DIR = Path(__file__).parents[1] / "Dataset Round 01" / "Cleaned"
    RANDOM_STATE = int(hash("uBIw")) % 2**32

    np.random.seed(RANDOM_STATE)
    return DATA_DIR, RANDOM_STATE


@app.cell
def _():
    # ATTRIBUTES = [
    #     "ID",
    #     "Trial",
    #     "P3M",
    #     "P1M",
    #     "Brand_Likability",
    #     "Fre#visit",
    #     "PPA",
    #     "Spending",
    #     "NPS#P3M",
    # ]

    # segmentation_df = (
    #     pl.read_csv(DATA_DIR / "Brand Health.csv")
    #     .filter(pl.col("Brand") == "Highlands Coffee")
    #     .select(ATTRIBUTES)
    # )

    # CUSTOMER_ATTRIBUTES = [
    #     "ID",
    #     "City",
    #     "Group_size",
    #     "Age",
    #     "MPI#Mean",
    #     # "MostFavourite",
    #     # "Gender",
    #     # "Occupation",
    #     "Occupation#group",
    # ]
    # segmentation_df = segmentation_df.join(
    #     pl.read_csv(DATA_DIR / "customer.csv").select(CUSTOMER_ATTRIBUTES),
    #     on="ID",
    # ).to_dummies(
    #     [
    #         "City",
    #         # "MostFavourite",
    #         # "Gender",
    #         # "Occupation",
    #         "Occupation#group",
    #     ]
    # )

    # segmentation_df = (
    #     segmentation_df.join(
    #         pl.read_csv(DATA_DIR / "Needstate.csv")
    #         .select(
    #             [
    #                 "ID",
    #                 # "Needstates",
    #                 "NeedstateGroup",
    #             ]
    #         )
    #         .to_dummies(
    #             [
    #                 # "Needstates",
    #                 "NeedstateGroup"
    #             ]
    #         ),
    #         on="ID",
    #     )
    #     .group_by(segmentation_df.columns)
    #     .max()
    # )

    # segmentation_df = (
    #     segmentation_df.join(
    #         pl.read_csv(DATA_DIR / "companion.csv")
    #         .select(["ID", "Companion#group"])
    #         .to_dummies("Companion#group"),
    #         on="ID",
    #     )
    #     .group_by(segmentation_df.columns)
    #     .max()
    # )

    # segmentation_df
    return


@app.cell
def _(DATA_DIR, cs, pl):
    ATTRIBUTES = [
        "ID",
        "Trial",
        "P3M",
        "P1M",
        "Brand_Likability",
        "Fre#visit",
        "PPA",
        "Spending",
        "NPS#P3M",
    ]

    segmentation_df = (
        pl.read_csv(DATA_DIR / "Brand Health.csv")
        .filter(pl.col("Brand") == "Highlands Coffee")
        .select(ATTRIBUTES)
    )

    CUSTOMER_ATTRIBUTES = [
        "ID",
        "City",
        "Group_size",
        "Age",
        "MPI#Mean",
        # "MostFavourite",
        # "Gender",
        # "Occupation",
        "Occupation#group",
    ]
    segmentation_df = segmentation_df.join(
        pl.read_csv(DATA_DIR / "customer.csv").select(CUSTOMER_ATTRIBUTES),
        on="ID",
    )

    segmentation_df = (
        segmentation_df.join(
            pl.read_csv(DATA_DIR / "Needstate.csv").select(
                [
                    "ID",
                    # "Needstates",
                    "NeedstateGroup",
                ]
            ),
            on="ID",
        )
        .group_by(segmentation_df.columns)
        .max()
    )

    segmentation_df = (
        segmentation_df.join(
            pl.read_csv(DATA_DIR / "companion.csv").select(
                ["ID", "Companion#group"]
            ),
            on="ID",
        )
        .group_by(segmentation_df.columns)
        .max()
    )

    segmentation_df = segmentation_df.cast(
        {cs.integer(): pl.Float64, cs.boolean(): pl.Float64}
    )
    segmentation_df
    return (segmentation_df,)


@app.cell
def _(FAMD, segmentation_df):
    famd_df = segmentation_df.drop("ID").to_pandas()
    eigen_summary = FAMD(40).fit(famd_df).eigenvalues_summary
    n_components = (eigen_summary["eigenvalue"].astype(float) > 1).sum()
    n_components

    famd = FAMD(n_components).fit_transform(famd_df)
    famd
    return famd, n_components


@app.cell
def _(
    KElbowVisualizer,
    KMeans,
    RANDOM_STATE,
    famd,
    n_components,
    pl,
    segmentation_df,
):
    model = KMeans(random_state=RANDOM_STATE)
    visualizer = KElbowVisualizer(
        model, k=(2, n_components + 1), metric="silhouette", timings=False
    )
    visualizer.fit(famd)
    visualizer.show()

    n_clusters = visualizer.elbow_value_
    segments = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE).fit_predict(
        famd
    )

    segmented_df = segmentation_df.hstack(
        [
            pl.Series("Segment", segments).map_elements(
                lambda x: f"Segment {x + 1}", return_dtype=str
            )
        ]
    ).sort("Segment")
    segmented_df
    return (segmented_df,)


@app.cell
def _(pl, plt, segmented_df):
    segment_info = (
        segmented_df.drop("ID")
        .to_dummies(
            ["City", "Occupation#group", "NeedstateGroup", "Companion#group"]
        )
        .group_by("Segment")
        .agg(
            [
                pl.count("Segment").alias("Segment Size"),
                pl.col("*").mean().round(2),
            ]
        )
        .sort("Segment")
    )

    plt.pie(
        segment_info["Segment Size"],
        labels=segment_info["Segment"],
        autopct="%.0f%%",
    )
    plt.title("Segment's Proportion")
    plt.show()
    segment_info
    return (segment_info,)


@app.cell
def _(go, segment_info):
    theta = segment_info["Segment"].to_list()
    theta.append(theta[0])

    fig_perceptual = go.Figure()
    for perceptual in ["Trial", "P3M", "P1M", "Brand_Likability", "NPS#P3M"]:
        p = segment_info[perceptual] / segment_info[perceptual].max()
        p = p.to_list()
        p.append(p[0])
        fig_perceptual.add_trace(
            go.Scatterpolar(
                r=p,
                theta=theta,
                name=perceptual,
            )
        )

    fig_perceptual.update_layout(title=dict(text="Perceptual Factors"))
    fig_perceptual.write_image("Perceptual.png")
    return (theta,)


@app.cell
def _(go, segment_info, theta):
    fig_behavioral = go.Figure()
    for behavioral in ["Fre#visit", "PPA", "Spending"]:
        b = segment_info[behavioral] / segment_info[behavioral].max()
        b = b.to_list()
        b.append(b[0])
        fig_behavioral.add_trace(
            go.Scatterpolar(
                r=b,
                theta=theta,
                name=behavioral,
            )
        )

    fig_behavioral.update_layout(title=dict(text="Behavioral Factors"))
    fig_behavioral.write_image("Behavioral.png")
    return


@app.cell
def _(cs, plt, segment_info):
    fig_needstate, axs_needstate = plt.subplots(2, 2, figsize=(16, 9))
    fig_needstate.suptitle("Needstate Distribution")

    needstate_info = (
        segment_info.select(cs.starts_with("Needstate")) * 100
    ).transpose(include_header=True)

    for idx in range(4):
        row = idx // 2
        col = idx % 2
        axs_needstate[row][col].pie(
            needstate_info[f"column_{idx}"],
            autopct="%.0f%%",
        )
        axs_needstate[row][col].set_title(f"Segment {idx + 1}")

    fig_needstate.legend(
        needstate_info["column"].to_list(),
        loc="center",
    )
    plt.show()


    fig_com, axs_com = plt.subplots(2, 2, figsize=(16, 9))
    fig_com.suptitle("Companion Group Distribution")

    com_info = (segment_info.select(cs.starts_with("Companion")) * 100).transpose(
        include_header=True
    )

    for idx in range(4):
        row = idx // 2
        col = idx % 2
        axs_com[row][col].pie(
            com_info[f"column_{idx}"],
            autopct="%.0f%%",
        )
        axs_com[row][col].set_title(f"Segment {idx + 1}")

    fig_com.legend(
        com_info["column"].to_list(),
        loc="center",
    )
    plt.show()


    fig_city, axs_city = plt.subplots(2, 2, figsize=(16, 9))
    fig_city.suptitle("City Distribution")

    city_info = (segment_info.select(cs.starts_with("City")) * 100).transpose(
        include_header=True
    )

    for idx in range(4):
        row = idx // 2
        col = idx % 2
        axs_city[row][col].pie(
            city_info[f"column_{idx}"],
            autopct="%.0f%%",
        )
        axs_city[row][col].set_title(f"Segment {idx + 1}")

    fig_city.legend(
        city_info["column"].to_list(),
        loc="center",
    )
    plt.show()


    fig_occ, axs_occ = plt.subplots(2, 2, figsize=(16, 9))
    fig_occ.suptitle("Occupation Distribution")

    occ_info = (segment_info.select(cs.starts_with("Occupation")) * 100).transpose(
        include_header=True
    )

    for idx in range(4):
        row = idx // 2
        col = idx % 2
        axs_occ[row][col].pie(
            occ_info[f"column_{idx}"],
            autopct="%.0f%%",
        )
        axs_occ[row][col].set_title(f"Segment {idx + 1}")

    fig_occ.legend(
        occ_info["column"].to_list(),
        loc="center",
    )
    plt.show()
    return


@app.cell
def _(plt, segmented_df, sns):
    sns.boxplot(data=segmented_df, x="Segment", y="Age")
    plt.title("Age distribution")
    return


@app.cell
def _(plt, segmented_df, sns):
    sns.boxplot(data=segmented_df, x="Segment", y="MPI#Mean", log_scale=True)
    plt.title("MPI distribution")
    return


@app.cell
def _(plt, segmented_df, sns):
    sns.boxplot(data=segmented_df, x="Segment", y="Group_size")
    plt.title("Group size distribution")
    return


if __name__ == "__main__":
    app.run()
