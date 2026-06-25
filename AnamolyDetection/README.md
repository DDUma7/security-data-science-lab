# Anomaly Detection using Clustering Algorithms

This project performs anomaly detection on network-intrusion data using clustering algorithms.

## Main Notebook

- `notebooks/anomaly_detection_using_clustering_algorithms.ipynb`

## Dataset

The notebook uses:

- `data/raw/Network_Intrusion_Detection/Train_data.csv`

The `class` column is not used during clustering. It is used after clustering only for evaluation.

## Algorithms

- KMeans
- Agglomerative Clustering
- DBSCAN

## Included Analysis

- Dataset loading and EDA
- Class distribution
- Protocol, service, and flag distribution
- Sampling and preprocessing
- PCA visualization
- KMeans elbow curve and silhouette curve
- Agglomerative clustering dendrogram
- DBSCAN k-distance plot
- DBSCAN eps tuning
- Internal clustering metrics: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
- External evaluation metrics: Accuracy, Weighted F1, Anomaly F1
- Confusion matrices
- Classification reports
- Final model comparison

## Run Again

From the repository root:

```bash
.venv/bin/python AnamolyDetection/scripts/create_anomaly_clustering_notebook.py
MPLCONFIGDIR=/tmp .venv/bin/jupyter nbconvert --to notebook --execute --inplace AnamolyDetection/notebooks/anomaly_detection_using_clustering_algorithms.ipynb
```
