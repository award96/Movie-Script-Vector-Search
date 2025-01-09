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
    html_template = """
    <html>
    <head>
        <title>Movie Script Embedded Vector Similarity</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/select2/4.0.13/js/select2.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/select2/4.0.13/css/select2.min.css"/>

        <style>
            body {
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
            }
            .sidebar {
                background-color: #f0f0f0;
                width: 200px;
                height: 100vh; /* full viewport height */
                padding: 20px;
                float: left;
            }
            .content {
                margin-left: 220px; /* leave space for the sidebar */
                padding: 20px;
            }
            .plot-container {
                margin: 0 auto;   /* center the container */
                margin-left: 100px; /* leave space for the sidebar */
                margin-right: 100px; /* looks more centered */
                // max-width: 800px; /* limit the plot width */
                margin-bottom: 30px; /* spacing between plots */
            }
        </style>
    </head>
    <body>
        <!-- Sidebar -->
        <div class="sidebar">
            <h3>Navigation</h3>
            <ul style="list-style: none; padding-left: 0;">
                <li><a href="/">Home</a></li>
                <li><a href="/other_plots">Other Plots</a></li>
            </ul>
        </div>
        
        <!-- Main Content Area -->
        <div class="content">
            <h1 style="text-align: center;">Movie Script Embedded Vector Similarity</h1>
            <div style="margin: 20px; text-align: center;">
                <label for="movie-select">Choose a Movie:</label>
                <select id="movie-select" style="width: 300px;"></select>
                <button onclick="loadVisuals()">Load Visualizations</button>
            </div>
            
            <!-- Plot containers (narrow) -->
            <div class="plot-container">
                <h2 style="text-align: left;">Similarity Metrics Correlation</h2>
                <div id="correlation"></div>
            </div>
            <div class="plot-container">
                <h2 style="text-align: left;">Nearest Neighbors</h2>
                <div id="neighbors"></div>
            </div>
        </div>

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
    """Simple placeholder route for other plots."""
    return """
    <html>
    <head><title>Other Plots</title></head>
    <body>
        <h1>Other Plots Page</h1>
        <p>This is where you can add or showcase different plots.</p>
        <a href="/">Return to Home</a>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True)
