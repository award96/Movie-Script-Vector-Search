import polars as pl
import torch
from IPython.display import display

def return_matches(
    movie_title: str,
    similarity_name_value_pairs: dict, # dict mapping "Distance": torch.Tensor 
    #                                               (n_movies, n_movies) ...
    movie_dataset: pl.DataFrame,
    metric: str = "Distance"
) -> pl.DataFrame:
    """Get the titles of movies with the most similar script to the input 
    tile according to the metric calculated over the embeddings.

    Args:
        movie_title (str): The movie script embedding to rank similarity against.
        similarity_name_value_pairs (dict[str: torch.Tensor]): A dictionary mapping
            similarity metrics (ie "Distance") to torch.Tensors of shape
            (n_movies, n_movies) which are all such comparisons of that metric.
            These tensors should have the same index as movie_dataset.
        movie_dataset (pl.DataFrame): The movie names, scripts, years, etc
            with the same index as embeddings.
        metric (str): The similarity metric to rank on. One of "Distance", 
            "Dotproduct", "Cosine".
    Returns:
        pl.DataFrame of other movies ranked by metric over script similarity.
    """
    # we will use index as PK
    index_in_df: int = get_index_in_df_from_title(movie_title, movie_dataset)
    # create a dataframe where the columns are similarity metrics
    # and the rows are other movies
    df = create_similarity_df(
        similarity_name_value_pairs,
        movie_dataset,
        index_in_df
    ).filter(
        pl.col("index") != index_in_df # omit self similarity
    ).sort(
        pl.col(metric),
        descending = metric != "Distance" # distance is asc, rest are desc
    )
    return df

def calculate_similarity_pairs_for_index(
    embeddings: torch.Tensor,
    idx: int
) -> dict[torch.Tensor]:
    """For the embeddings tensor of shape (n_movies, hidden_state_size)
    calculate the dotproduct, cosine similarity, and distance between the
    idx hidden state and all hidden states.

    Args:
        embeddings (torch.Tensor): The script embeddings of shape 
            (n_movies, hidden_state_size).
        idx (int): The index of the hidden state we want to calculate
            comparisons with.
    Returns:
        dict where the keys are the names of metrics, and the values
        are torch vectors of shape (hidden_state_size,).
    """
    # (hidden_state_size, )
    dot = torch.tensordot(embeddings[idx], embeddings, dims=([0], [1]))
    
    cos_sim_layer = torch.nn.CosineSimilarity(dim=1)
    # (hidden_state_size, )
    cos = cos_sim_layer(embeddings[idx], embeddings)
    # (hidden_state_size, )
    distances = torch.cdist(embeddings[idx].unsqueeze(0), embeddings).squeeze(0)

    return { # all same length
        "Dotproduct": dot,
        "Cosine": cos,
        "Distance": distances
    }

def calculate_all_similarity_pairs(
    embeddings: torch.Tensor
) -> dict[torch.Tensor]:
    """For the embeddings tensor of shape (n_movies, hidden_state_size)
    calculate the dotproduct, cosine similarity, and distance between the
    all hidden states.

    Args:
        embeddings (torch.Tensor): The script embeddings of shape 
            (n_movies, hidden_state_size).
    Returns:
        dict where the keys are the names of metrics, and the values
        are torch tensors of shape (n_movies, hidden_state_size).
    """
    ## Dot Product
    # (n_movies, hidden_state_size )
    dot = torch.tensordot(embeddings, embeddings, dims=([1], [1]))
    
    ## Cosine
    cos_sim_layer = torch.nn.CosineSimilarity(dim=2)
    # create square tensor for n_movies squared cosine comparisons
    # (n_movies, n_movies, hidden_state_size)
    embeddings_duplicated_over_0 = embeddings.unsqueeze(1).expand(-1, embeddings.shape[0], -1)
    embeddings_duplicated_over_1 = embeddings.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
    # for reference:
    # embeddings_duplicated_over_0[specific_index, any_index, :] == embeddings[specific_index, :]
    # embeddings_duplicated_over_1[any_index, specific_index, :] == embeddings[specific_index, :]

    # (n_movies, hidden_state_size )
    cos = cos_sim_layer(embeddings_duplicated_over_0, embeddings_duplicated_over_1)

    ## Distance (KNN)
    # (n_movies, hidden_state_size )
    distances = torch.cdist(embeddings, embeddings).squeeze(0)
    
    return { # all same shape
        "Dotproduct": dot,
        "Cosine": cos,
        "Distance": distances
    }

def create_similarity_df(
    similarity_name_value_pairs: dict,
    movie_dataset: pl.DataFrame,
    index: int
) -> pl.DataFrame:
    """Using the dictionary keys as column names, and values at index
    as columns, create a dataframe, join it to the movie dataset
    on the index, and return it.

    Args:
        similarity_name_value_pairs (dict[str: torch.Tensor]): A dictionary mapping
            similarity metrics (ie "Distance") to torch.Tensors of shape
            (n_movies, n_movies) which are all such comparisons of that metric.
            These tensors should have the same index as movie_dataset. Use 
            torch.unsqueeze(0) and index=0 to make this function compatible 
            with an array of shape (n_movies,).
        movie_dataset (pl.DataFrame): The movie data with the same index as the
            torch arrays in similarity_name_value_pairs.values().
    Returns:
        pl.DataFrame of the similarity_name_value_pairs joined to the movie dataset.
    """

    return pl.DataFrame(
        {
            measure_name: torch_comparison_tensor[index].numpy() for \
            measure_name, torch_comparison_tensor in similarity_name_value_pairs.items()
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
    """End user will pick based off title, so this function
    converts that title into a shared index. The index is shared by the
    movie_dataset and the embeddings.

    Args:
        movie_title (str): The movie title we want the index of.
        movie_dataset (pl.DataFrame): The movie titles in the column "movie_title".
    Returns:
        int: The index of movie_title in movie_dataset["movie_title"].
    """
    idx = movie_dataset.filter(pl.col("movie_title").str.to_lowercase() == movie_title.lower()).select(pl.col("index"))
    if len(idx) < 1:
        raise ValueError(f"Could not find movie {movie_title}")
    elif len(idx) > 1:
        raise ValueError(f"Found multiple matches for movie {movie_title}\n{idx}")
    else:
        idx = idx[0, "index"]
        return idx

def display_top_n_matches(
    movie_title: str,
    embeddings: torch.Tensor,
    movie_dataset: pl.DataFrame,
    n: int = 5,
    metric: str = "Distance"
) -> None:
    df = return_matches(movie_title, embeddings, movie_dataset, metric)
    print(movie_title)
    display(df[:n])
    print("\nCorrelations:")
    display_similarity_correlation(df, df.select(pl.col(pl.Float32, pl.Float64)).columns)

def display_similarity_correlation(
    data: pl.DataFrame,
    columns: list[str]
) -> None:
    corr = data.select(pl.col(columns)).corr()
    display(corr)

if __name__ == "__main__":
    embeddings = torch.load("data/out/scripts-embedded.pt", weights_only=True)
    movie_data = pl.read_parquet("data/out/movie-script-dataset.parquet")
    similarity_name_value_pairs = calculate_all_similarity_pairs(embeddings)
    rnd_movie = movie_data.sample(1)[0, "movie_title"]
    print(rnd_movie)
    display(
        return_matches(
            rnd_movie,
            similarity_name_value_pairs,
            movie_data
        )
    )