import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def read_data(file_path):
    """Reads the CSV file and returns the adjusted close prices."""
    df_raw = pd.read_csv(file_path)
    adj_close = df_raw.iloc[:, -1]
    return adj_close


def calculate_log_returns(adj_close):
    """Calculates log returns from adjusted close prices."""
    log_ret = np.log(adj_close / adj_close.shift(1)).fillna(0).to_numpy()
    return pd.DataFrame(data=log_ret, columns=["returns"])


def jump_condition(data, std_threshold):
    return (
        (data["returns"] > data["ma"] + std_threshold * data["volatility"])
        | (data["returns"] < data["ma"] - std_threshold * data["volatility"])
    ).astype(int)


def autocorr(data, window, autocorr_lag):
    return (
        data["returns"]
        .rolling(window)
        .apply(lambda x: x.autocorr(lag=autocorr_lag), raw=False)
    )


def engineer_features(returns, window=20, std_threshold=2, autocorr_lag=2):
    data = returns.copy()
    data["volatility"] = data["returns"].rolling(window).std()
    data["ma"] = data["returns"].rolling(window).mean()
    data["skew"] = data["returns"].rolling(window).skew()
    data["kurt"] = data["returns"].rolling(window).kurt()
    data["autocorr"] = autocorr(data, window, autocorr_lag)
    data["jump"] = jump_condition(data, std_threshold)
    return data.dropna()  # Start from the correct index after rolling operations


def scale(data):
    """Scales the features using StandardScaler."""
    scaler = StandardScaler()
    binary_ommitted_df = data[[column for column in data.columns if column != "jump"]]
    scaled_data = scaler.fit_transform(binary_ommitted_df)
    jump_feature = np.array(data["jump"]).reshape(-1, 1)
    return np.concatenate((scaled_data, jump_feature), axis=1)


def kmeans_cluster(data, scaled_data, n_clusters, random_state):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(scaled_data)
    data["kmeans_regime"] = kmeans.labels_


def hierarchical_cluster(data, scaled_data, n_clusters):
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    agg_clustering.fit(scaled_data)
    data["hierarchical_regime"] = agg_clustering.labels_


def spectral_cluster(data, scaled_data, n_clusters, affinity):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity)
    spectral.fit(scaled_data)
    data["spectral_regime"] = spectral.labels_


def cluster_absolute_averages(data):
    regimes = [f"{i}_regime" for i in ["kmeans", "hierarchical", "spectral"]]
    cluster_abs_avgs = {regime: data.groupby(regime).mean() for regime in regimes}
    return cluster_abs_avgs


def mass_cluster(data, scaled_data, n_clusters, random_state, affinity):
    features = ["returns", "volatility", "ma", "skew", "autocorr", "jump"]
    kmeans_cluster(data, scaled_data, n_clusters, random_state)
    hierarchical_cluster(data, scaled_data, n_clusters)
    spectral_cluster(data, scaled_data, n_clusters, affinity)
    cluster_abs_avgs = {
        key.replace("_regime", ""): value[features]
        for key, value in cluster_absolute_averages(data).items()
    }
    return data, cluster_abs_avgs


def plot_cluster_characteristics(cluster_rel_avgs):
    # Determine the number of cluster methods and set grid size accordingly
    cluster_methods = [
        label.replace("_regime", "") for label in cluster_rel_avgs.keys()
    ]
    n_methods = len(cluster_methods)

    # Create subplots with a number of rows and columns based on the number of methods
    nrows = (n_methods + 1) // 2  # Use integer division to determine rows
    ncols = (
        2 if n_methods > 1 else 1
    )  # Set columns to 2, or 1 if there's only one method

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 6))
    axes = axes.flatten()  # Flatten to easily iterate over axes

    for i, ax in enumerate(axes):
        if i < n_methods:  # Check if there are methods left to plot
            cluster_method = cluster_methods[i]
            cluster_rel_avgs[cluster_method].plot(kind="bar", ax=ax)
            ax.set_xticklabels(cluster_rel_avgs[cluster_method].index, rotation=0)
            ax.set_title(f"{cluster_method.title()} Cluster Characteristics")
            ax.set_xlabel("Regimes")
            ax.grid(True)
        else:
            ax.axis("off")  # Hide any unused axes

    plt.tight_layout()
    plt.savefig("meme.png")


def fmt(value, fmt_type):
    """Formats the score values for display."""
    if fmt_type == "dcm":
        return f"{value:.2f}"
    return value


def score_clusters(data):
    """Scores the clusters using Silhouette and Calinski-Harabasz scores."""
    cluster_labels = [column for column in data.columns if "regime" in column]
    scores = []
    for label in cluster_labels:
        cluster_data = data[
            data[label] != -1
        ]  # Exclude noise points if any (specifically for DBSCAN).
        if cluster_data.empty:
            continue
        labels = cluster_data[label]
        X = cluster_data.drop(cluster_labels, axis=1)
        silhouette = silhouette_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        scores.append(
            [
                label.title().replace("_", "").replace("Regime", ""),
                fmt(silhouette, "dcm"),
                fmt(calinski_harabasz, "dcm"),
            ]
        )

    header = ["Method", "Silhouette Score", "Calinski-Harabasz Score"]
    df = pd.DataFrame(scores, columns=header)
    return df.set_index("Method").sort_values(ascending=False, by="Silhouette Score")


def plot_price_chart(data, adj_close):
    """Plots the price chart with highlighted regimes."""
    plt.figure(figsize=(60, 40))
    plt.plot(
        adj_close.index,
        adj_close,
        label="Adjusted Close Price",
        color="blue",
        alpha=0.5,
    )

    # Define a color scheme for each clustering method
    color_map = {
        "kmeans": "lightcoral",
        "hierarchical": "lightgreen",
        "spectral": "lightblue",
    }

    # Plot regimes with shaded areas
    for label in ["kmeans_regime", "hierarchical_regime", "spectral_regime"]:
        if label in data.columns:
            unique_labels = data[label].unique()
            for regime in unique_labels:
                regime_data = data[data[label] == regime]
                if not regime_data.empty:
                    plt.axvspan(
                        regime_data.index[0],
                        regime_data.index[-1],
                        color=color_map[label.replace("_regime", "")],
                        label=f"{label.replace('_regime', '').title()} Regime {regime}"
                        if regime == unique_labels[0]
                        else "",
                        alpha=0.3,
                    )  # Adjust alpha for shading

    plt.title("Adjusted Close Price with Highlighted Regimes")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("price_chart.png")


def plot_log_return_chart(data):
    """Plots the log return chart with highlighted regimes."""
    plt.figure(figsize=(60, 40))
    plt.plot(
        data.index, data["returns"], label="Log Returns", color="orange", alpha=0.5
    )

    # Define a color scheme for each clustering method
    color_map = {
        "kmeans": "lightcoral",
        "hierarchical": "lightgreen",
        "spectral": "lightblue",
    }

    # Plot regimes with shaded areas
    for label in ["kmeans_regime", "hierarchical_regime", "spectral_regime"]:
        if label in data.columns:
            unique_labels = data[label].unique()
            for regime in unique_labels:
                regime_data = data[data[label] == regime]
                if not regime_data.empty:
                    plt.axvspan(
                        regime_data.index[0],
                        regime_data.index[-1],
                        color=color_map[label.replace("_regime", "")],
                        label=f"{label.replace('_regime', '').title()} Regime {regime}"
                        if regime == unique_labels[0]
                        else "",
                        alpha=0.3,
                    )  # Adjust alpha for shading

    plt.title("Log Returns with Highlighted Regimes")
    plt.xlabel("Date")
    plt.ylabel("Log Returns")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("log_return_chart.png")


def main():
    # Set parameters for clustering
    file_path = "/home/LilAiluropoda/Projects/fyp_project/app/data/stock_data/AAPL/AAPL19972007.csv"
    window = 20
    std_threshold = 2
    autocorr_lag = 2
    n_clusters = 2
    random_state = 42
    affinity = "nearest_neighbors"

    # Read and process data
    adj_close = read_data(file_path)
    returns = calculate_log_returns(adj_close)
    data = engineer_features(returns, window, std_threshold, autocorr_lag)
    scaled_data = scale(data)

    # Perform clustering
    data, cluster_abs_avgs = mass_cluster(
        data, scaled_data, n_clusters, random_state, affinity
    )

    # Plot cluster characteristics
    plot_cluster_characteristics(cluster_abs_avgs)

    # Score the clusters
    scores = score_clusters(data)
    print(scores)

    # Plot price and log return charts with regimes highlighted
    plot_price_chart(data, adj_close)
    plot_log_return_chart(data)


if __name__ == "__main__":
    main()
