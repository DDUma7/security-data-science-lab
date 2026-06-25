from pathlib import Path

import nbformat as nbf


OUTPUT_DIR = Path("AnamolyDetection/notebooks")
OUTPUT_PATH = OUTPUT_DIR / "anomaly_detection_using_clustering_algorithms.ipynb"


COMMON_IMPORTS = r'''
import os
from pathlib import Path
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    precision_recall_fscore_support,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
SAMPLE_PER_CLASS = 1500
BW_HATCHES = ["///", "\\\\\\", "xx", "..", "++", "oo"]
BW_COLORS = ["white", "#E6E6E6", "#BFBFBF", "#8C8C8C", "#666666", "#D9D9D9"]
BW_MARKERS = ["o", "s", "^", "D", "x", "P", "*"]
BW_LINESTYLES = ["-", "--", "-.", ":"]

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
})
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_colwidth", 120)
'''.strip()


HELPERS = r'''
def add_bar_labels(ax, horizontal=False):
    for patch in ax.patches:
        value = patch.get_width() if horizontal else patch.get_height()
        if np.isnan(value):
            continue
        if horizontal:
            ax.annotate(
                f"{value:,.0f}",
                (value, patch.get_y() + patch.get_height() / 2),
                ha="left",
                va="center",
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=9,
            )
        else:
            ax.annotate(
                f"{value:,.0f}",
                (patch.get_x() + patch.get_width() / 2, value),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=9,
            )


def plot_count_bar(counts, title, xlabel="", ylabel="", horizontal=False, top_n=None):
    plot_counts = counts.copy()
    if top_n is not None:
        plot_counts = plot_counts.head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(plot_counts))) if horizontal else (8, 4.5))
    colors = [BW_COLORS[i % len(BW_COLORS)] for i in range(len(plot_counts))]
    if horizontal:
        bars = ax.barh(plot_counts.index.astype(str), plot_counts.values, color=colors, edgecolor="black", linewidth=1.0)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.margins(x=0.15)
    else:
        bars = ax.bar(plot_counts.index.astype(str), plot_counts.values, color=colors, edgecolor="black", linewidth=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.margins(y=0.15)
    for idx, bar in enumerate(bars):
        bar.set_hatch(BW_HATCHES[idx % len(BW_HATCHES)])
    ax.set_title(title)
    add_bar_labels(ax, horizontal=horizontal)
    plt.tight_layout()
    plt.show()


def get_data_root():
    candidates = [
        Path("AnamolyDetection/data"),
        Path("../data"),
        Path.cwd() / "AnamolyDetection" / "data",
    ]
    return next(path for path in candidates if path.exists())


def stratified_sample(data, label_col, n_per_class=SAMPLE_PER_CLASS):
    parts = []
    for label_value, group in data.groupby(label_col):
        parts.append(group.sample(min(len(group), n_per_class), random_state=RANDOM_STATE))
    return pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def preprocess_features(data, label_col):
    feature_data = data.drop(columns=[label_col]).copy()
    categorical_cols = feature_data.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    processed = pd.get_dummies(feature_data, columns=categorical_cols, drop_first=False)
    processed = processed.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(processed)
    return processed, scaled, scaler


def map_clusters_to_anomaly(cluster_labels, y_true):
    mapped = np.zeros_like(y_true, dtype=int)
    mapping_rows = []
    for cluster_id in sorted(pd.Series(cluster_labels).unique()):
        mask = cluster_labels == cluster_id
        if cluster_id == -1:
            mapped_label = 1
            rule = "DBSCAN noise -> anomaly"
        else:
            counts = pd.Series(y_true[mask]).value_counts()
            mapped_label = int(counts.idxmax())
            rule = "majority label for evaluation"
        mapped[mask] = mapped_label
        mapping_rows.append(
            {
                "cluster": cluster_id,
                "mapped_prediction": "anomaly" if mapped_label == 1 else "normal",
                "cluster_size": int(mask.sum()),
                "true_anomaly_rate": float(y_true[mask].mean()) if mask.sum() else np.nan,
                "rule": rule,
            }
        )
    return mapped, pd.DataFrame(mapping_rows)


def internal_cluster_metrics(X, cluster_labels, ignore_noise=True):
    labels = np.asarray(cluster_labels)
    if ignore_noise:
        valid_mask = labels != -1
        X_eval = X[valid_mask]
        labels_eval = labels[valid_mask]
    else:
        X_eval = X
        labels_eval = labels

    unique_labels = np.unique(labels_eval)
    if X_eval.shape[0] < 2 or len(unique_labels) < 2:
        return {
            "Silhouette Score": np.nan,
            "Davies-Bouldin Index": np.nan,
            "Calinski-Harabasz Index": np.nan,
            "Metric Notes": "Not enough non-noise clusters",
        }

    return {
        "Silhouette Score": silhouette_score(X_eval, labels_eval),
        "Davies-Bouldin Index": davies_bouldin_score(X_eval, labels_eval),
        "Calinski-Harabasz Index": calinski_harabasz_score(X_eval, labels_eval),
        "Metric Notes": "Noise ignored" if ignore_noise and np.any(labels == -1) else "All points used",
    }


def evaluate_clustering(name, cluster_labels, y_true, X_for_internal_metrics):
    y_pred, mapping_df = map_clusters_to_anomaly(cluster_labels, y_true)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    result = {
        "Model": name,
        "Clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
        "Noise Points": int((cluster_labels == -1).sum()),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Weighted Precision": precision,
        "Weighted Recall": recall,
        "Weighted F1": f1,
        "Anomaly F1": f1_score(y_true, y_pred, zero_division=0),
    }
    result.update(internal_cluster_metrics(X_for_internal_metrics, cluster_labels, ignore_noise=True))

    print(f"{name} cluster-to-label mapping")
    display(mapping_df)

    print(f"{name} classification report")
    display(pd.DataFrame(classification_report(y_true, y_pred, target_names=["normal", "anomaly"], output_dict=True, zero_division=0)).T.round(4))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "anomaly"])
    disp.plot(cmap="Greys", values_format="d", colorbar=False)
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.show()
    return result, y_pred


def plot_pca_scatter(pca_df, color_col, title):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = sorted(pca_df[color_col].unique(), key=lambda x: str(x))
    for idx, label_value in enumerate(labels):
        subset = pca_df[pca_df[color_col] == label_value]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=18,
            facecolors="none" if idx % 2 == 0 else BW_COLORS[idx % len(BW_COLORS)],
            edgecolors="black",
            marker=BW_MARKERS[idx % len(BW_MARKERS)],
            linewidths=0.7,
            alpha=0.75,
            label=str(label_value),
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title=color_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_metric_lines(df, x_col, y_cols, title, ylabel):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for idx, col in enumerate(y_cols):
        ax.plot(
            df[x_col],
            df[col],
            color=BW_COLORS[min(idx + 1, len(BW_COLORS) - 1)] if idx else "black",
            marker=BW_MARKERS[idx % len(BW_MARKERS)],
            linestyle=BW_LINESTYLES[idx % len(BW_LINESTYLES)],
            linewidth=1.8,
            label=col,
        )
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    plt.show()
'''.strip()


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(source):
    return nbf.v4.new_code_cell(source.strip())


def build_notebook():
    cells = [
        markdown(
            """
# Anomaly Detection using Clustering Algorithms

This notebook performs anomaly detection on network-intrusion data using three unsupervised clustering algorithms:

- KMeans
- Agglomerative Clustering
- DBSCAN

The data is taken from the existing `AnamolyDetection/data` folder. The `class` label is **not used for clustering**; it is used only after clustering to evaluate how well each algorithm separated normal and anomalous traffic.

Plotting note: charts use grayscale, markers, and hatching so they remain readable in black-and-white xerox copies.
"""
        ),
        markdown("## 1. Imports and Settings"),
        code(COMMON_IMPORTS),
        markdown("## 1.1 Helper Functions"),
        code(HELPERS),
        markdown("## 2. Load Dataset"),
        code(
            r'''
DATA_ROOT = get_data_root()
raw_path = DATA_ROOT / "raw" / "Network_Intrusion_Detection" / "Train_data.csv"

df_raw = pd.read_csv(raw_path)
print("Using dataset:", raw_path.resolve())
print("Raw shape:", df_raw.shape)
display(df_raw.head())
'''
        ),
        markdown("## 3. Exploratory Data Analysis"),
        code(
            r'''
print("Columns:", df_raw.columns.tolist())
print("\nMissing values")
display(pd.DataFrame({
    "missing": df_raw.isna().sum(),
    "missing_pct": (df_raw.isna().mean() * 100).round(2),
}).sort_values("missing", ascending=False).head(15))

class_counts = df_raw["class"].value_counts()
display(class_counts.rename_axis("class").reset_index(name="count"))
plot_count_bar(class_counts, title="Normal vs Anomaly Class Distribution", ylabel="Records")

for col in ["protocol_type", "flag", "service"]:
    print(f"Top values for {col}")
    display(df_raw[col].value_counts().head(12).rename_axis(col).reset_index(name="count"))
    plot_count_bar(df_raw[col].value_counts(), title=f"Top {col} Values", xlabel="Records", horizontal=True, top_n=12)
'''
        ),
        markdown("## 4. Sampling and Preprocessing"),
        code(
            r'''
df = stratified_sample(df_raw, label_col="class", n_per_class=SAMPLE_PER_CLASS)
print("Sampled shape:", df.shape)
display(df["class"].value_counts().rename_axis("class").reset_index(name="count"))

y_true = (df["class"] != "normal").astype(int).to_numpy()
X_processed, X_scaled, scaler = preprocess_features(df, label_col="class")

print("Processed feature shape:", X_processed.shape)
display(X_processed.head())
'''
        ),
        markdown("## 5. PCA Visualization"),
        code(
            r'''
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["true_label"] = np.where(y_true == 1, "anomaly", "normal")

print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))
plot_pca_scatter(pca_df, color_col="true_label", title="PCA View of True Normal vs Anomaly Labels")
'''
        ),
        markdown("## 6. KMeans Clustering"),
        code(
            r'''
k_values = range(2, 11)
kmeans_rows = []
for k in k_values:
    model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels = model.fit_predict(X_scaled)
    kmeans_rows.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette": silhouette_score(X_scaled, labels),
    })

kmeans_curve = pd.DataFrame(kmeans_rows)
display(kmeans_curve.round(4))
plot_metric_lines(kmeans_curve, "k", ["inertia"], "KMeans Elbow Curve", "Inertia")
plot_metric_lines(kmeans_curve, "k", ["silhouette"], "KMeans Silhouette by k", "Silhouette")

kmeans_model = KMeans(n_clusters=2, n_init=20, random_state=RANDOM_STATE)
kmeans_labels = kmeans_model.fit_predict(X_scaled)
pca_df["KMeans_cluster"] = kmeans_labels.astype(str)
plot_pca_scatter(pca_df, color_col="KMeans_cluster", title="KMeans Clusters on PCA View")

results = []
predictions = {}
kmeans_result, kmeans_pred = evaluate_clustering("KMeans", kmeans_labels, y_true, X_scaled)
results.append(kmeans_result)
predictions["KMeans"] = kmeans_pred
'''
        ),
        markdown("## 7. Agglomerative Clustering"),
        code(
            r'''
dendrogram_sample_size = min(200, X_scaled.shape[0])
rng = np.random.default_rng(RANDOM_STATE)
dendrogram_idx = rng.choice(X_scaled.shape[0], size=dendrogram_sample_size, replace=False)
linked = linkage(X_scaled[dendrogram_idx], method="ward")

plt.figure(figsize=(12, 5))
dendrogram(linked, truncate_mode="level", p=4, color_threshold=0, above_threshold_color="black")
plt.title("Hierarchical Clustering Dendrogram Sample")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

agg_model = AgglomerativeClustering(n_clusters=2, linkage="ward")
agg_labels = agg_model.fit_predict(X_scaled)
pca_df["Agglomerative_cluster"] = agg_labels.astype(str)
plot_pca_scatter(pca_df, color_col="Agglomerative_cluster", title="Agglomerative Clusters on PCA View")

agg_result, agg_pred = evaluate_clustering("Agglomerative", agg_labels, y_true, X_scaled)
results.append(agg_result)
predictions["Agglomerative"] = agg_pred
'''
        ),
        markdown("## 8. DBSCAN Hyperparameter Exploration"),
        code(
            r'''
min_samples = 10
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors.fit(X_scaled)
distances, _ = neighbors.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])

plt.figure(figsize=(9, 4.8))
plt.plot(np.arange(len(k_distances)), k_distances, color="black", linewidth=1.5)
plt.title(f"DBSCAN k-distance Plot (min_samples={min_samples})")
plt.xlabel("Sorted points")
plt.ylabel("Distance to kth nearest neighbor")
plt.tight_layout()
plt.show()

eps_candidates = np.quantile(k_distances, [0.75, 0.85, 0.90, 0.95, 0.98])
dbscan_rows = []
for eps in eps_candidates:
    labels = DBSCAN(eps=float(eps), min_samples=min_samples, n_jobs=-1).fit_predict(X_scaled)
    pred, _ = map_clusters_to_anomaly(labels, y_true)
    dbscan_rows.append({
        "eps": float(eps),
        "clusters": len(set(labels)) - (1 if -1 in labels else 0),
        "noise_points": int((labels == -1).sum()),
        "noise_pct": (labels == -1).mean() * 100,
        "accuracy": accuracy_score(y_true, pred),
        "anomaly_f1": f1_score(y_true, pred, zero_division=0),
        **internal_cluster_metrics(X_scaled, labels, ignore_noise=True),
    })

dbscan_tuning = pd.DataFrame(dbscan_rows)
display(dbscan_tuning.round(4))
plot_metric_lines(dbscan_tuning, "eps", ["accuracy", "anomaly_f1"], "DBSCAN eps Tuning", "Score")
plot_metric_lines(dbscan_tuning, "eps", ["noise_pct"], "DBSCAN Noise Percentage by eps", "Noise %")
'''
        ),
        markdown("## 9. DBSCAN Clustering"),
        code(
            r'''
best_eps = dbscan_tuning.sort_values("anomaly_f1", ascending=False).iloc[0]["eps"]
print("Selected eps from tuning table:", round(best_eps, 4))

dbscan_model = DBSCAN(eps=float(best_eps), min_samples=min_samples, n_jobs=-1)
dbscan_labels = dbscan_model.fit_predict(X_scaled)
pca_df["DBSCAN_cluster"] = dbscan_labels.astype(str)
plot_pca_scatter(pca_df, color_col="DBSCAN_cluster", title="DBSCAN Clusters and Noise on PCA View")

dbscan_result, dbscan_pred = evaluate_clustering("DBSCAN", dbscan_labels, y_true, X_scaled)
results.append(dbscan_result)
predictions["DBSCAN"] = dbscan_pred
'''
        ),
        markdown("## 10. Model Comparison"),
        code(
            r'''
comparison_df = pd.DataFrame(results)
display(comparison_df.round(4))

print("Internal clustering metrics")
internal_metrics_df = comparison_df[
    [
        "Model",
        "Clusters",
        "Noise Points",
        "Silhouette Score",
        "Davies-Bouldin Index",
        "Calinski-Harabasz Index",
        "Metric Notes",
    ]
]
display(internal_metrics_df.round(4))

print("External evaluation metrics using known labels")
external_metrics_df = comparison_df[
    [
        "Model",
        "Accuracy",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1",
        "Anomaly F1",
    ]
]
display(external_metrics_df.round(4))

plot_count_bar(
    comparison_df.set_index("Model")["Anomaly F1"],
    title="Clustering Algorithm Comparison by Anomaly F1",
    ylabel="Anomaly F1",
)

plot_count_bar(
    comparison_df.set_index("Model")["Weighted F1"],
    title="Clustering Algorithm Comparison by Weighted F1",
    ylabel="Weighted F1",
)

plot_count_bar(
    comparison_df.set_index("Model")["Silhouette Score"],
    title="Clustering Algorithm Comparison by Silhouette Score",
    ylabel="Silhouette Score",
)

plot_count_bar(
    comparison_df.set_index("Model")["Davies-Bouldin Index"],
    title="Clustering Algorithm Comparison by Davies-Bouldin Index",
    ylabel="Davies-Bouldin Index",
)
'''
        ),
        markdown("## 11. Final Prediction Table"),
        code(
            r'''
prediction_table = pd.DataFrame({
    "actual": np.where(y_true == 1, "anomaly", "normal"),
    "KMeans": np.where(predictions["KMeans"] == 1, "anomaly", "normal"),
    "Agglomerative": np.where(predictions["Agglomerative"] == 1, "anomaly", "normal"),
    "DBSCAN": np.where(predictions["DBSCAN"] == 1, "anomaly", "normal"),
})
display(prediction_table.head(20))
'''
        ),
        markdown(
            """
## Conclusion

This project used three clustering algorithms for anomaly detection on network-intrusion data. KMeans and Agglomerative Clustering were evaluated by mapping clusters to the known normal/anomaly labels after clustering. DBSCAN was evaluated by treating noise points as anomalies and mapping remaining clusters for comparison.

The notebook reports both internal clustering metrics and external evaluation metrics. Internal metrics such as Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index judge cluster compactness and separation without using labels. External metrics such as Accuracy, Weighted F1, and Anomaly F1 use the known `class` labels only for evaluation.

In a real deployment, labels would not be available during clustering. The labels here are used only to measure how well the unsupervised clusters correspond to known network anomalies.
"""
        ),
    ]

    nb = nbf.v4.new_notebook(cells=cells)
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return nb


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbf.write(notebook, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
