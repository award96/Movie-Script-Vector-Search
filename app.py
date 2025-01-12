from flask import Flask, render_template_string, request, jsonify
import polars as pl
import plotly.express as px
import torch
import similarity
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
# so we can pre-load all 110^2 comparisons
# for Distance, Dotproduct, and Cosine
# {"Distance": torch.Tensor of shape (n_movies, n_movies) ...}
similarity_name_value_pairs: dict[torch.Tensor] = \
    similarity.calculate_all_similarity_pairs(
        torch.load("data/out/scripts-embedded.pt", weights_only=True)
    )

# randomly sorted titles for
# user to select from
titles_list = movie_dataset.select(pl.col("movie_title").shuffle())\
    ["movie_title"].to_list()
movie_titles = [{"id": t, "text": t} for t in titles_list]

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
def visualize():
    data = request.get_json()
    movie_title = data.get('movie_title')

    if not movie_title:
        return jsonify({"error": "No movie title provided"}), 400

    # Get all movies sorted by distance
    neighbors_df = similarity.return_matches(
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

    return render_template_string(html_template) # , movie_titles_json=movie_titles_json)

if __name__ == '__main__':
    app.run(debug=True)
