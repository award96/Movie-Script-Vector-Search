from flask import Flask, render_template_string, request, jsonify
import polars as pl
import plotly.express as px
import pandas as pd
import similarity
import json

# Load Data
embeddings = similarity.load_embedding_tensors()
movie_dataset = similarity.load_movie_dataset()

# Convert polars DF to a simple list of titles
# NOTE: .select(...).shuffle() returns a DataFrame with one column; 
# you’ll need to access it as a Series then convert to a Python list.
df_titles = movie_dataset.select(pl.col("movie_title").shuffle())
titles_list = df_titles["movie_title"].to_list()

# Optionally, transform into the {id, text} shape if needed by select2
# (Otherwise, a plain list of strings can also work depending on your config)
movie_titles = [{"id": t, "text": t} for t in titles_list]

metric = "Distance"

# Flask App Initialization
app = Flask(__name__)

# Route for Initial Page
@app.route('/')
def index():
    # Create JSON for movie titles that we can safely embed in the template
    movie_titles_json = json.dumps(movie_titles)
    html_template = """
    <html>
        <head>
            <title>Movie Script Similarity</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/select2/4.0.13/js/select2.min.js"></script>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/select2/4.0.13/css/select2.min.css"/>
        </head>
        <body>
            <h1 style="text-align: center;">Movie Script Similarity</h1>
            <div style="margin: 20px; text-align: center;">
                <label for="movie-select">Choose a Movie:</label>
                <select id="movie-select" style="width: 300px;"></select>
                <button onclick="loadVisuals()">Load Visualizations</button>
            </div>
            <h2>Similarity Metrics Correlation</h2>
            <div id="correlation"></div>
            <h2>Top Nearest Neighbors</h2>
            <div id="neighbors"></div>

            <script>
                // Parse the server-passed JSON movie titles
                var movieData = {{ movie_titles_json | safe }};

                // Initialize Select2 Dropdown
                $(document).ready(function() {
                    $('#movie-select').select2({
                        placeholder: "Select a movie",
                        data: movieData
                    });
                });

                function loadVisuals() {
                    const selectedMovie = $('#movie-select').val();
                    if (!selectedMovie) {
                        alert("Please select a movie.");
                        return;
                    }
                    $.ajax({
                        url: '/visualize',
                        type: 'POST',
                        contentType: 'application/json',
                        data: JSON.stringify({ movie_title: selectedMovie }),
                        success: function(response) {
                            // response.corr_plot and response.neighbors_plot are strings,
                            // so convert them to an object first:
                            var corrPlot = JSON.parse(response.corr_plot);
                            var neighborsPlot = JSON.parse(response.neighbors_plot);

                            Plotly.newPlot('correlation', corrPlot.data, corrPlot.layout);
                            Plotly.newPlot('neighbors', neighborsPlot.data, neighborsPlot.layout);
                        }
                    });
                }
            </script>
        </body>
    </html>
    """
    return render_template_string(html_template, movie_titles_json=movie_titles_json)

# Route for Generating Visualizations
@app.route('/visualize', methods=['POST'])
def visualize():
    data = request.get_json()
    movie_title = data.get('movie_title')

    if not movie_title:
        return jsonify({"error": "No movie title provided"}), 400

    # Recalculate data based on selected movie
    neighbors_df = similarity.return_matches(movie_title, embeddings, movie_dataset, metric)

    correlation_df = neighbors_df.select(pl.col(pl.Float32, pl.Float64)).corr()

    neighbors_df = neighbors_df.to_pandas()
    correlation_df = correlation_df.to_pandas()

    # Create correlation heatmap
    fig_corr = px.imshow(
        correlation_df,
        labels=dict(x="Metric", y="Metric", color="Correlation"),
        x=["Dotproduct", "Cosine", "Distance"],
        y=["Dotproduct", "Cosine", "Distance"],
        color_continuous_scale="Portland"
    )
    fig_corr.update_layout(title="Similarity Metrics Correlation Heatmap")

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
    fig_neighbors.update_layout(title="Top N Nearest Neighbors Visualization")

    return jsonify({
        "corr_plot": fig_corr.to_json(),
        "neighbors_plot": fig_neighbors.to_json()
    })


if __name__ == '__main__':
    app.run(debug=True)
