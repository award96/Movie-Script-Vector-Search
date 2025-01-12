import joblib
import traceback
import random
import plotly.express as px
import polars as pl
import pandas as pd
import torch

__reducer = joblib.load("data/out/UMAP-reducer.joblib")
__label_movies = [
    '12 Angry Men',
    'Aladdin',
    'Avatar',
    'Avengers: Endgame',
    'Barbie',
    'Black Panther Wakanda Forever',
    'Cidade de Deus (City of God)',
    'The Dark Knight',
    'Fight Club',
    'Forrest Gump',
    "The Bourne Identity",
    'Frozen (Disney)',
    'The Godfather',
    'Nope',
    'The LEGO Movie',
    'Coraline',
    'Harry Potter and the Half Blood Prince',
    'Inception',
    'The Shawshank Redemption',
    'Interstellar',
    "It's a Wonderful Life",
    'The Lord of the Rings The Fellowship of the Ring',
    'The Matrix',
    'Star Wars Episode V The Empire Strikes Back',
    'The Nightmare Before Christmas',
    'Pulp Fiction',
    'Saving Private Ryan',
    'Wall-E',
    'Up'
]
franchises = [
    "Bourne",
    "Alien",
    "Batman",
    "Lord of the Rings",
    "Godfather",
    "Inside Out",
    "Shrek",
    "Top Gun"
]

def make_all_visualizations(
    emblow: pl.DataFrame
) -> None:
    try:
        return [
            franchise_scatter_plot(emblow),
            genre_scatter_plot(emblow),
            selected_labels_scatter_plot(emblow)
        ]
    except Exception as e:
        print(emblow)
        raise e
def franchise_scatter_plot(emblow: pl.DataFrame):
    fig = px.scatter(
        emblow,
        x='x',
        y='y',
        color='Franchise',
        symbol='Franchise',
        size = 'point_size',
        text='franchise_label',
        hover_name = "movie_title",
        hover_data =["year", "genre", "script_length"],
        title = "UMAP of Embedded Scripts by Franchise",
        width = int(1.25e3),
        height = int(0.65e3)
    )
    fig.update_layout(
        font=dict(
            family='Arial Black', 
            size=14
        ),
        title_font = dict(
            family='Arial Black', 
            size=24
        )
    )
    return fig

def genre_scatter_plot(emblow: pl.DataFrame, focus: str = "Comedy") -> None:

    fig = px.scatter(
        emblow.with_columns(
            pl.col("genre").str.to_lowercase().str.contains(focus.lower(), literal=True)
            .alias(focus)
        ).with_columns(
            pl.when(pl.col(focus))
            .then(1.25)
            .otherwise(1)
            .alias(focus + "size")
        ),
        x='x',
        y='y',
        color=focus,
        symbol=focus,
        size = [2]*len(emblow), # focus + "size",
        # text = f'{focus}_label',
        hover_name = "movie_title",
        hover_data =["year", "genre", focus],
        title = "UMAP of Embedded Scripts by Genre",
        width = int(1.25e3),
        height = int(0.65e3)
    )

    fig.update_layout(
        font=dict(
            family='Arial Black', 
            size=14
        ),
        title_font = dict(
            family='Arial Black', 
            size=24
        )
    )

    return fig

def selected_labels_scatter_plot(
    emblow: pl.DataFrame
) -> object:
    fig = px.scatter(
        emblow,
        x="x", 
        y="y",
        text = "selected_labels",
        hover_name = "movie_title",
        hover_data =["year", "genre", "script_length"],
        title = "UMAP of Embedded Scripts. Selected movies labeled",
        width = int(1.25e3),
        height = int(0.65e3)
    )
    fig.update_xaxes(
        range=[-17,-11.5]
    )

    fig.update_layout(
        font=dict(
            family='Arial Black', 
            size=12
        ),
        title_font = dict(
            family='Arial Black', 
            size=24
        )
    )
    return fig

def reduce_data_and_add_vis_cols(
    embeddings: torch.Tensor, 
    movie_dataset: pl.DataFrame
) -> pl.DataFrame:
    emblow = pd.DataFrame(__reducer.transform(embeddings), columns = ['x', 'y'])
    emblow = (
        pl.from_pandas(emblow).lazy()
        .with_row_index()
        .join(movie_dataset.lazy(), "index", "inner")
        .with_columns(
            # label some of the movies on the plot
            selected_labels = pl.when(pl.col("movie_title").is_in(__label_movies))
            .then(pl.col("movie_title"))
            .otherwise(pl.lit(None)),
            # assign franchise
            Franchise = pl.col("movie_title").str.replace("The Dark Knight", "Batman")
            .map_elements(
                lambda x: max([
                    f if (f in x) else ".No sequels included" for f in franchises
                ]), 
                return_dtype=pl.String)
        )
    )

    # for plotting
    # only display one label per franchise
    indexed_unique_labels = (
        emblow.with_columns(
            # label the franchise as text
            franchise_label = pl.when(pl.col("Franchise") != ".No sequels included")
                .then(pl.col("Franchise"))
                .otherwise(pl.lit(None))
        )
        # remove duplicates (too messy for plotting since they are close together)
        .unique(pl.col("franchise_label"), keep='any', maintain_order=False)
        .filter(pl.col("franchise_label").is_not_null())
        .select(pl.col("index", "franchise_label"))
    )
    # join back just the first label for each franchise
    emblow = (
        emblow.join(
            indexed_unique_labels,
            on = "index",
            how = "left",
            validate="1:1"
        )
        .with_columns(
            # make the franchise points bigger on the plot
            point_size =  pl.when(pl.col("Franchise") != ".No sequels included")
            .then(pl.lit(3))
            .otherwise(pl.lit(1))
        )
        .with_columns(
            (
                pl.col("genre").str.to_lowercase().str.contains(focus.lower(), literal=True)
                    .alias(focus)
                for focus in emblow.select(pl.col("genre").str.split(",").flatten()).unique().collect()["genre"]
                if len(focus or "") > 0
            )
        )
    # execute the query plan
    ).collect()

    return emblow


if __name__ == "__main__":
    
    movie_dataset = (
        pl.scan_parquet("data/out/movie-script-dataset.parquet")
        .with_columns( script_length = pl.col("script").str.len_chars() )
        .select(pl.col("index", "movie_title", "genre", "script_length", "year"))
        .collect()
    )
    
    embeddings = torch.load("data/out/scripts-embedded.pt", weights_only=True)
    # embeddings low dimension
    emblow = reduce_data_and_add_vis_cols(embeddings, movie_dataset)
    all_vis = make_all_visualizations(emblow)
    [figure.show() for figure in all_vis]