from flask import Flask, render_template_string
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import similarity

embeddings = similarity.load_embedding_tensors()
movie_dataset = similarity.load_movie_dataset()
movie_title = movie_dataset.sample(1)[0, "movie_title"]
metric = "Distance"
# Simulated Data (Replace with your actual Polars DataFrame imports)
neighbors_df = similarity.return_matches(movie_title, embeddings, movie_dataset, metric)
correlation_df = neighbors_df.select(pl.col(pl.Float32, pl.Float64)).corr()

neighbors_df = neighbors_df.to_pandas()
correlation_df = correlation_df.to_pandas()

# Flask App Initialization
app = Flask(__name__)

@app.route('/')
def index():
    print(movie_title)
    # Heatmap for correlation
    fig_corr = px.imshow(
        correlation_df,
        labels=dict(x="Metric", y="Metric", color="Correlation"),
        x=["Dotproduct", "Cosine similarity", "Distance"],
        y=["Dotproduct", "Cosine similarity", "Distance"],
        color_continuous_scale="RdBu"
    )
    fig_corr.update_layout(title="Similarity Metrics Correlation Heatmap")

    # Scatter plot for neighbors
    fig_neighbors = px.scatter(
        neighbors_df,
        x="Distance",
        y="Dotproduct",
        text="movie_title",
        color="Cosine similarity",
        color_continuous_scale="Blues",
        size="Dotproduct",
        hover_data=["index", "Dotproduct", "Cosine similarity", "Distance", "movie_title"]
    )
    fig_neighbors.update_traces(textposition='top center')
    fig_neighbors.update_layout(title="Top N Nearest Neighbors Visualization")

    # Render Template
    html_template = """
    <html>
        <head>
            <title>Nearest Neighbors Visualization</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Similarity Metrics Correlation</h1>
            <div id="correlation"></div>
            <h1>Top Nearest Neighbors</h1>
            <div id="neighbors"></div>
            <script>
                var corrPlot = {{corr_plot | safe}};
                Plotly.newPlot('correlation', corrPlot.data, corrPlot.layout);

                var neighborsPlot = {{neighbors_plot | safe}};
                Plotly.newPlot('neighbors', neighborsPlot.data, neighborsPlot.layout);
            </script>
        </body>
    </html>
    """
    return render_template_string(
        html_template,
        corr_plot=fig_corr.to_json(),
        neighbors_plot=fig_neighbors.to_json()
    )

if __name__ == '__main__':
    app.run(debug=True)
