from flask import Flask, render_template_string, request, jsonify
import polars as pl
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import torch
import vector_similarity
import make_other_plots
import json

# load the movie dataset
# omitting the actual scripts
movie_dataset = (
    pl.scan_parquet("data/out/movie-script-dataset.parquet")
      .with_columns( script_length = pl.col("script").str.len_chars() )
      .select(pl.col("index", "movie_title", "genre", "script_length", "year"))
      .collect()
)

# There are only 110 movie script embeddings
embeddings: torch.Tensor = torch.load("data/out/scripts-embedded.pt", weights_only=True)
# so we can pre-load all 110^2 comparisons
# for Distance, Dotproduct, and Cosine
# {"Distance": torch.Tensor of shape (n_movies, n_movies) ...}
similarity_name_value_pairs: dict[torch.Tensor] = \
    vector_similarity.calculate_all_similarity_pairs(embeddings)

umap_2d_embeddings: pl.DataFrame = make_other_plots.reduce_data_and_add_vis_cols(embeddings, movie_dataset)

# randomly sorted titles for
# user to select from
titles_list = movie_dataset.select(pl.col("movie_title").shuffle())\
    ["movie_title"].to_list()
movie_titles = [{"id": t, "text": t} for t in titles_list]
genres_list = (
    movie_dataset.lazy()
    .select(pl.col("genre").str.split(",").flatten())
    .unique()
    .collect()
    ["genre"]
    .to_list()
)
genres = [{"id": g, "text": g} for g in genres_list]

# currently not variable
metric = "Distance"

# --------------------------
# Flask Application
# --------------------------
app = Flask(__name__)

@app.route('/')
def index():
    # Create JSON for movie titles that we can safely embed in the template
    movie_titles_json = json.dumps(movie_titles)
    with open('html/index.html', 'r') as file:
       html_template = file.read()
    return render_template_string(html_template, movie_titles_json=movie_titles_json)

# Route for Generating Visualizations
@app.route('/visualize', methods=['POST'])
def make_plots():
    data = request.get_json()
    movie_title = data.get('movie_title')

    if not movie_title:
        return jsonify({"error": "No movie title provided"}), 400

    # Get all movies sorted by distance
    neighbors_df = vector_similarity.return_matches(
        movie_title, 
        similarity_name_value_pairs, 
        movie_dataset, 
        metric
    )

    correlation_df = neighbors_df.select(pl.col(pl.Float32, pl.Float64)).corr()

    # Create correlation heatmap
    fig_corr = px.imshow(
        correlation_df,
        labels=dict(x="Metric", y="Metric", color="Correlation"),
        x=["Dotproduct", "Cosine", "Distance"],
        y=["Dotproduct", "Cosine", "Distance"],
        color_continuous_scale="Portland"
    )
    fig_corr.update_layout(
        title=f"Similarity Metrics Correlation Heatmap<br>for {movie_title}",
        # width=700,
        # height=500,
        # margin=dict(l=50, r=50, b=50, t=50)
    )

    # Create scatter plot for neighbors
    fig_neighbors = px.scatter(
        neighbors_df,
        x="Distance",
        y="Dotproduct",
        text="movie_title",
        color="Cosine",
        color_continuous_scale="Blues",
        size="Dotproduct",
        hover_data=["Dotproduct", "Cosine", "Distance", "movie_title"]
    )
    fig_neighbors.update_traces(textposition='top center')
    fig_neighbors.update_layout(
        title=f"Nearest Neighbors to {movie_title}"
    )

    return jsonify({
        "corr_plot": fig_corr.to_json(),
        "neighbors_plot": fig_neighbors.to_json()
    })

@app.route('/other_plots')
def other_plots():
    with open('html/other_plots.html', 'r') as file:
       html_template = file.read()

    # List of Plotly Figure objects
    plotly_figs = make_other_plots.make_all_visualizations(umap_2d_embeddings)  
    figs_json = [json.dumps(fig, cls=PlotlyJSONEncoder) for fig in plotly_figs]
    genres_json = json.dumps(genres)

    return render_template_string(html_template, figs=figs_json, genres=genres_json)

@app.route('/update_genre_plot', methods=['POST'])
def update_genre_plot():
    data = request.get_json()
    selected_genre = data.get('genre')

    # Generate the updated figure
    # Re-run some portion of make_other_plots or just call genre_scatter_plot
    # But note that genre_scatter_plot picks a random focus from the splitted genres.
    # Let’s make a small refactor to accept `focus` from user:
    updated_fig = make_other_plots.genre_scatter_plot(umap_2d_embeddings, selected_genre)

    return jsonify(json.dumps(updated_fig, cls=PlotlyJSONEncoder))

if __name__ == '__main__':
    app.run(debug=True)
