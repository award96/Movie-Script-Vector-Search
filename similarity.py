import polars as pl
import torch
from IPython.display import display

def return_matches(
    movie_title: str,
    embeddings: torch.Tensor,
    movie_dataset: pl.DataFrame,
    metric: str = "Distance"
) -> pl.DataFrame:
    index_in_df: int = get_index_in_df_from_title(movie_title, movie_dataset)
    similarity_name_value_pairs: dict = calculate_similarity_pairs_for_index(
        embeddings,
        index_in_df
    )
    df = create_similarity_df(
        similarity_name_value_pairs,
        movie_dataset
    ).filter(pl.col("index") != index_in_df).sort(pl.col(metric)) # omit self similarity
    return df

def display_top_n_matches(
    movie_title: str,
    embeddings: torch.Tensor,
    movie_dataset: pl.DataFrame,
    n: int = 5,
    metric: str = "Distance"
) -> None:
    df = return_matches(movie_title, embeddings, movie_dataset, metric)
    print(movie_title)
    display_similarity_correlation(df, df.select(pl.col(pl.Float32, pl.Float64)).columns)
    display(df[:n])


def calculate_similarity_pairs_for_index(
    embeddings: torch.Tensor,
    idx: int
) -> dict[torch.Tensor]:
    dot = torch.tensordot(embeddings[idx], embeddings, dims=([0], [1]))
    
    cos_sim_layer = torch.nn.CosineSimilarity(dim=1)
    cos = cos_sim_layer(embeddings[idx], embeddings)

    distances = torch.cdist(embeddings[idx].unsqueeze(0), embeddings).squeeze(0)

    return {
        "Dotproduct": dot,
        "Cosine similarity": cos,
        "Distance": distances
    }

def calculate_all_similarity_pairs(): pass

def create_similarity_df(
    similarity_name_value_pairs: dict,
    movie_dataset: pl.DataFrame
) -> pl.DataFrame:

    return pl.DataFrame(
        {
            measure_name: torch_array.numpy() for \
            measure_name, torch_array in similarity_name_value_pairs.items()
        }
    ).with_row_index()\
        .join(
            movie_dataset.select(pl.col("index", "movie_title")),
            on = pl.col("index"),
            how = "inner"
        )

def get_index_in_df_from_title(
        movie_title: str,
        movie_dataset: pl.DataFrame
    ) -> int:
    idx = movie_dataset.filter(pl.col("movie_title").str.to_lowercase() == movie_title.lower()).select(pl.col("index"))
    if len(idx) < 1:
        raise ValueError(f"Could not find movie {movie_title}")
    elif len(idx) > 1:
        raise ValueError(f"Found multiple matches for movie {movie_title}\n{idx}")
    else:
        idx = idx[0, "index"]
        return idx
    
def display_similarity_correlation(
    data: pl.DataFrame,
    columns: list[str]
) -> None:
    corr = data.select(pl.col(columns)).corr()
    display(corr)

def load_embedding_tensors() -> torch.Tensor:
    return torch.load("data/out/scripts-embedded.pt", weights_only=True)
def load_movie_dataset() -> pl.DataFrame:
    return pl.read_parquet("data/out/movie-script-dataset.parquet")

if __name__ == "__main__":
    embeddings = load_embedding_tensors()
    movie_data = load_movie_dataset()
    display_top_n_matches(
        movie_data.sample(1)[0, "movie_title"],
        embeddings,
        movie_data,
        n=5
    )