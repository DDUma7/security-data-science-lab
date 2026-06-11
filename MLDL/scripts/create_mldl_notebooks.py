from __future__ import annotations

import csv
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
NOTEBOOK_DIR = ROOT / "notebooks"

KERNEL_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def write_notebook(path: Path, cells: list):
    nb = nbf.v4.new_notebook(cells=cells, metadata=KERNEL_METADATA)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


def convert_naive_bayes_xlsx():
    """Convert the tiny course XLSX sample without requiring openpyxl."""
    source = DATA_DIR / "naive_bayes_purchase.xlsx"
    target = DATA_DIR / "naive_bayes_purchase.csv"
    if not source.exists() or target.exists():
        return

    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(source) as zf:
        shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        shared_strings = [
            "".join(node.itertext())
            for node in shared_root.findall("x:si", ns)
        ]
        sheet_root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    rows = []
    for row in sheet_root.findall(".//x:sheetData/x:row", ns):
        values = []
        for cell in row.findall("x:c", ns):
            raw = cell.findtext("x:v", default="", namespaces=ns)
            value = shared_strings[int(raw)] if cell.attrib.get("t") == "s" and raw else raw
            values.append(value)
        rows.append(values)

    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)


SETUP = r'''
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

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120})

DATA_DIR = Path("../data")
if not DATA_DIR.exists():
    DATA_DIR = Path("MLDL/data")

pd.set_option("display.max_columns", 120)
'''


def linear_regression():
    return [
        md("""
        # Linear Regression

        Predict company profit from R&D spend, administration cost, marketing spend, and state.
        This notebook modernizes the course sample by using a scikit-learn `Pipeline` instead of
        deprecated manual one-hot encoding.
        """),
        code(SETUP),
        code(r'''
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv(DATA_DIR / "1000_companies.csv", encoding="utf-8-sig")
display(df.head())
display(df.describe(include="all"))
'''),
        code(r'''
target = "Profit"
X = df.drop(columns=[target])
y = df[target]

numeric_features = ["R&D Spend", "Administration", "Marketing Spend"]
categorical_features = ["State"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("regressor", LinearRegression()),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

metrics = pd.DataFrame(
    {
        "metric": ["MAE", "RMSE", "R2"],
        "value": [
            mean_absolute_error(y_test, predictions),
            np.sqrt(mean_squared_error(y_test, predictions)),
            r2_score(y_test, predictions),
        ],
    }
)
display(metrics)
'''),
        code(r'''
feature_names = model.named_steps["preprocess"].get_feature_names_out()
coef_df = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": model.named_steps["regressor"].coef_,
    }
).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
display(coef_df)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_test, predictions, edgecolor="black", alpha=0.75)
lims = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
axes[0].plot(lims, lims, color="black", linestyle="--")
axes[0].set_title("Actual vs Predicted Profit")
axes[0].set_xlabel("Actual")
axes[0].set_ylabel("Predicted")

residuals = y_test - predictions
axes[1].scatter(predictions, residuals, edgecolor="black", alpha=0.75)
axes[1].axhline(0, color="black", linestyle="--")
axes[1].set_title("Residuals")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual - Predicted")
plt.tight_layout()
plt.show()
'''),
    ]


def logistic_regression():
    return [
        md("""
        # Logistic Regression

        Binary classification on the breast cancer diagnosis CSV from the course archive.
        The model estimates the probability of malignant vs benign diagnosis from numeric cell measurements.
        """),
        code(SETUP),
        code(r'''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv(DATA_DIR / "breast_cancer_logistic.csv")
df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")])
display(df.head())
display(df["diagnosis"].value_counts().rename_axis("diagnosis").reset_index(name="count"))
'''),
        code(r'''
X = df.drop(columns=["id", "diagnosis"])
y = LabelEncoder().fit_transform(df["diagnosis"])  # B=0, M=1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

model = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ]
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"ROC AUC:  {roc_auc_score(y_test, proba):.3f}")
print(classification_report(y_test, pred, target_names=["benign", "malignant"]))
'''),
        code(r'''
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=["benign", "malignant"], ax=axes[0], cmap="Greys"
)
axes[0].set_title("Confusion Matrix")
RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1], color="black")
axes[1].set_title("ROC Curve")
plt.tight_layout()
plt.show()

coef_df = pd.DataFrame(
    {
        "feature": X.columns,
        "coefficient": model.named_steps["logreg"].coef_[0],
    }
).assign(abs_coefficient=lambda d: d["coefficient"].abs())
display(coef_df.sort_values("abs_coefficient", ascending=False).head(12))
'''),
    ]


def kmeans_clustering():
    return [
        md("""
        # K-Means Clustering

        Cluster cars from the course CSV using scaled numeric attributes. The notebook includes
        an elbow plot, silhouette scores, a 2D PCA view, and cluster profiles.
        """),
        code(SETUP),
        code(r'''
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(DATA_DIR / "cars_kmeans.csv", skipinitialspace=True)
df.columns = [c.strip() for c in df.columns]
display(df.head())
display(df.info())
'''),
        code(r'''
numeric_cols = ["mpg", "cylinders", "cubicinches", "hp", "weightlbs", "time-to-60", "year"]
X = df[numeric_cols].replace("?", np.nan).apply(pd.to_numeric, errors="coerce")

prep = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
)
X_scaled = prep.fit_transform(X)

scores = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    scores.append(
        {
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X_scaled, labels),
        }
    )

score_df = pd.DataFrame(scores)
display(score_df)
'''),
        code(r'''
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(score_df["k"], score_df["inertia"], marker="o", color="black")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")

axes[1].plot(score_df["k"], score_df["silhouette"], marker="o", color="black")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Score")
plt.tight_layout()
plt.show()

best_k = int(score_df.sort_values("silhouette", ascending=False).iloc[0]["k"])
print("Selected k from silhouette:", best_k)

kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE)
clusters = kmeans.fit_predict(X_scaled)
clustered = df.copy()
clustered["cluster"] = clusters

pca = PCA(n_components=2, random_state=RANDOM_STATE)
coords = pca.fit_transform(X_scaled)
plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
plot_df["cluster"] = clusters

plt.figure(figsize=(7, 5))
sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="cluster", palette="Set2", edgecolor="black")
plt.title("Car Clusters Projected with PCA")
plt.show()

display(clustered.groupby("cluster")[numeric_cols].mean().round(2))
'''),
    ]


def knn():
    return [
        md("""
        # K-Nearest Neighbors

        Predict diabetes outcome from the Pima diabetes CSV in the course archive. The notebook compares
        multiple `k` values after imputing invalid zero measurements and scaling features.
        """),
        code(SETUP),
        code(r'''
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(DATA_DIR / "diabetes_knn.csv")
display(df.head())
display(df["Outcome"].value_counts().rename_axis("Outcome").reset_index(name="count"))
'''),
        code(r'''
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
clean = df.copy()
clean[zero_as_missing] = clean[zero_as_missing].replace(0, np.nan)

X = clean.drop(columns=["Outcome"])
y = clean["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

results = []
for k in range(1, 26, 2):
    model = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results.append({"k": k, "accuracy": accuracy_score(y_test, pred), "f1": f1_score(y_test, pred)})

results_df = pd.DataFrame(results)
display(results_df)

best_k = int(results_df.sort_values(["f1", "accuracy"], ascending=False).iloc[0]["k"])
print("Selected k:", best_k)
'''),
        code(r'''
best_model = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=best_k)),
    ]
)
best_model.fit(X_train, y_train)
pred = best_model.predict(X_test)
print(classification_report(y_test, pred))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(results_df["k"], results_df["accuracy"], marker="o", label="accuracy", color="black")
axes[0].plot(results_df["k"], results_df["f1"], marker="s", label="f1", color="gray")
axes[0].set_xlabel("k")
axes[0].set_ylabel("score")
axes[0].set_title("KNN Model Selection")
axes[0].legend()
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=axes[1], cmap="Greys")
axes[1].set_title("Confusion Matrix")
plt.tight_layout()
plt.show()
'''),
    ]


def svm():
    return [
        md("""
        # Support Vector Machine

        The archive's SVM sample demonstrates a separating margin on generated blobs. This notebook keeps
        that visual idea and adds a real classification task using the breast cancer CSV.
        """),
        code(SETUP),
        code(r'''
from sklearn.datasets import make_blobs
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

X_blob, y_blob = make_blobs(n_samples=80, centers=2, cluster_std=1.1, random_state=6)
clf = SVC(kernel="linear", C=1000)
clf.fit(X_blob, y_blob)

plt.figure(figsize=(6, 5))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap="Set2", edgecolor="black")
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 80)
yy = np.linspace(ylim[0], ylim[1], 80)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors="black", levels=[-1, 0, 1], alpha=0.7, linestyles=["--", "-", "--"])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=120, facecolors="none", edgecolors="black")
plt.title("Linear SVM Margin")
plt.show()
'''),
        code(r'''
df = pd.read_csv(DATA_DIR / "breast_cancer_logistic.csv")
df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")])
X = df.drop(columns=["id", "diagnosis"])
y = LabelEncoder().fit_transform(df["diagnosis"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

pipe = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        ("svm", SVC(probability=True, random_state=RANDOM_STATE)),
    ]
)
search = GridSearchCV(
    pipe,
    param_grid={
        "svm__kernel": ["linear", "rbf"],
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale", 0.01],
    },
    scoring="f1",
    cv=5,
    n_jobs=-1,
)
search.fit(X_train, y_train)
pred = search.predict(X_test)

print("Best parameters:", search.best_params_)
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred, target_names=["benign", "malignant"]))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["benign", "malignant"], cmap="Greys")
plt.title("SVM Confusion Matrix")
plt.show()
'''),
    ]


def decision_tree():
    return [
        md("""
        # Decision Tree

        Train an interpretable tree on the course decision-tree CSV and inspect feature importance,
        the confusion matrix, and the top levels of the learned tree.
        """),
        code(SETUP),
        code(r'''
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv(DATA_DIR / "decision_tree_dataset.csv", encoding="utf-8-sig")
df = df.rename(columns={"Unnamed: 5": "target"})
df["target"] = df["target"].str.strip().str.lower()
display(df.head())
display(df["target"].value_counts().rename_axis("target").reset_index(name="count"))
'''),
        code(r'''
feature_cols = ["1", "2", "3", "4", "sum"]
X = df[feature_cols]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
tree = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=10,
    random_state=RANDOM_STATE,
)
tree.fit(X_train, y_train)
pred = tree.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred))
ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Greys")
plt.title("Decision Tree Confusion Matrix")
plt.show()

importance = pd.DataFrame(
    {"feature": feature_cols, "importance": tree.feature_importances_}
).sort_values("importance", ascending=False)
display(importance)
'''),
        code(r'''
plt.figure(figsize=(16, 7))
plot_tree(
    tree,
    feature_names=feature_cols,
    class_names=tree.classes_,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=9,
)
plt.title("Top Levels of the Decision Tree")
plt.show()
'''),
    ]


def random_forest():
    return [
        md("""
        # Random Forest

        Use the course iris CSV to train an ensemble of decision trees, then compare predictions and
        feature importances.
        """),
        code(SETUP),
        code(r'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree

df = pd.read_csv(DATA_DIR / "iris_random_forest.csv")
display(df.head())
display(df["Species"].value_counts().rename_axis("Species").reset_index(name="count"))
'''),
        code(r'''
X = df.drop(columns=["Id", "Species"])
labeler = LabelEncoder()
y = labeler.fit_transform(df["Species"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=RANDOM_STATE,
    oob_score=True,
)
forest.fit(X_train, y_train)
pred = forest.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"OOB score: {forest.oob_score_:.3f}")
print(classification_report(y_test, pred, target_names=labeler.classes_))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=labeler.classes_, cmap="Greys")
plt.xticks(rotation=20)
plt.title("Random Forest Confusion Matrix")
plt.show()
'''),
        code(r'''
importance = pd.DataFrame(
    {"feature": X.columns, "importance": forest.feature_importances_}
).sort_values("importance", ascending=False)
display(importance)

plt.figure(figsize=(7, 4))
sns.barplot(data=importance, x="importance", y="feature", color="gray", edgecolor="black")
plt.title("Random Forest Feature Importance")
plt.show()

plt.figure(figsize=(16, 7))
plot_tree(
    forest.estimators_[0],
    feature_names=X.columns,
    class_names=labeler.classes_,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=9,
)
plt.title("One Tree from the Forest")
plt.show()
'''),
    ]


def xgboost():
    return [
        md("""
        # XGBoost

        Train a boosted-tree classifier on the red wine quality CSV. If `xgboost` is installed,
        the notebook uses `XGBClassifier`; otherwise it falls back to scikit-learn's histogram gradient
        boosting so the workflow remains runnable offline.
        """),
        code(SETUP),
        code(r'''
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

df = pd.read_csv(DATA_DIR / "winequality_red.csv", sep=";")
df["good_quality"] = (df["quality"] >= 6).astype(int)
display(df.head())
display(df["good_quality"].value_counts().rename_axis("good_quality").reset_index(name="count"))
print("Using xgboost package:", HAS_XGBOOST)
'''),
        code(r'''
X = df.drop(columns=["quality", "good_quality"])
y = df["good_quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

if HAS_XGBOOST:
    model = XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
else:
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=15,
        random_state=RANDOM_STATE,
    )

model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print(type(model).__name__)
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"ROC AUC:  {roc_auc_score(y_test, proba):.3f}")
print(classification_report(y_test, pred, target_names=["lower quality", "good quality"]))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["lower", "good"], cmap="Greys")
plt.title("Boosted Tree Confusion Matrix")
plt.show()
'''),
        code(r'''
if HAS_XGBOOST:
    importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
else:
    result = permutation_importance(
        model, X_test, y_test, n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1
    )
    importance = pd.DataFrame(
        {"feature": X.columns, "importance": result.importances_mean}
    ).sort_values("importance", ascending=False)

display(importance)
plt.figure(figsize=(7, 4))
sns.barplot(data=importance.head(10), x="importance", y="feature", color="gray", edgecolor="black")
plt.title("Boosting Feature Importance")
plt.show()
'''),
    ]


MNIST_HELPERS = r'''
import gzip
import struct

def load_mnist_images(path, limit=None):
    with gzip.open(path, "rb") as fh:
        magic, count, rows, cols = struct.unpack(">IIII", fh.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number: {magic}")
        if limit is None:
            limit = count
        data = np.frombuffer(fh.read(rows * cols * limit), dtype=np.uint8)
    return data.reshape(limit, rows, cols).astype("float32") / 255.0

def load_mnist_labels(path, limit=None):
    with gzip.open(path, "rb") as fh:
        magic, count = struct.unpack(">II", fh.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number: {magic}")
        if limit is None:
            limit = count
        return np.frombuffer(fh.read(limit), dtype=np.uint8)
'''


def ann():
    return [
        md("""
        # Artificial Neural Network

        Train a dense neural network on the MNIST files from the course archive.
        The dataset is loaded locally, so no TensorFlow download is needed.
        """),
        code(SETUP + "\n" + MNIST_HELPERS),
        code(r'''
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

tf.random.set_seed(RANDOM_STATE)

train_images = load_mnist_images(DATA_DIR / "mnist/train-images-idx3-ubyte.gz", limit=8000)
train_labels = load_mnist_labels(DATA_DIR / "mnist/train-labels-idx1-ubyte.gz", limit=8000)
test_images = load_mnist_images(DATA_DIR / "mnist/t10k-images-idx3-ubyte.gz", limit=2000)
test_labels = load_mnist_labels(DATA_DIR / "mnist/t10k-labels-idx1-ubyte.gz", limit=2000)

print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
'''),
        code(r'''
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(
    train_images,
    train_labels,
    validation_split=0.15,
    epochs=3,
    batch_size=128,
    verbose=1,
)
'''),
        code(r'''
loss, acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test accuracy: {acc:.3f}")

pred = model.predict(test_images, verbose=0).argmax(axis=1)
print(classification_report(test_labels, pred))
ConfusionMatrixDisplay.from_predictions(test_labels, pred, cmap="Greys")
plt.title("ANN Confusion Matrix")
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(history.history["accuracy"], label="train", color="black")
plt.plot(history.history["val_accuracy"], label="validation", color="gray")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("ANN Learning Curve")
plt.legend()
plt.show()
'''),
    ]


def cnn():
    return [
        md("""
        # Convolutional Neural Network

        Train a small CNN on the local MNIST files from the course archive. CNN layers preserve image
        structure better than a dense ANN, which usually improves digit recognition.
        """),
        code(SETUP + "\n" + MNIST_HELPERS),
        code(r'''
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

tf.random.set_seed(RANDOM_STATE)

train_images = load_mnist_images(DATA_DIR / "mnist/train-images-idx3-ubyte.gz", limit=10000)[..., None]
train_labels = load_mnist_labels(DATA_DIR / "mnist/train-labels-idx1-ubyte.gz", limit=10000)
test_images = load_mnist_images(DATA_DIR / "mnist/t10k-images-idx3-ubyte.gz", limit=2000)[..., None]
test_labels = load_mnist_labels(DATA_DIR / "mnist/t10k-labels-idx1-ubyte.gz", limit=2000)
print(train_images.shape, test_images.shape)
'''),
        code(r'''
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(24, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(48, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    train_images,
    train_labels,
    validation_split=0.15,
    epochs=3,
    batch_size=128,
    verbose=1,
)
'''),
        code(r'''
loss, acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test accuracy: {acc:.3f}")
pred = model.predict(test_images, verbose=0).argmax(axis=1)
print(classification_report(test_labels, pred))
ConfusionMatrixDisplay.from_predictions(test_labels, pred, cmap="Greys")
plt.title("CNN Confusion Matrix")
plt.show()

sample_idx = np.arange(12)
fig, axes = plt.subplots(2, 6, figsize=(10, 4))
for ax, idx in zip(axes.ravel(), sample_idx):
    ax.imshow(test_images[idx].squeeze(), cmap="gray")
    ax.set_title(f"true {test_labels[idx]} / pred {pred[idx]}")
    ax.axis("off")
plt.tight_layout()
plt.show()
'''),
    ]


def lstm():
    return [
        md("""
        # Long Short-Term Memory Network

        Treat each MNIST image as a sequence of 28 rows with 28 features per timestep.
        This is a compact way to demonstrate LSTM sequence modeling with local course data.
        """),
        code(SETUP + "\n" + MNIST_HELPERS),
        code(r'''
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

tf.random.set_seed(RANDOM_STATE)

train_images = load_mnist_images(DATA_DIR / "mnist/train-images-idx3-ubyte.gz", limit=8000)
train_labels = load_mnist_labels(DATA_DIR / "mnist/train-labels-idx1-ubyte.gz", limit=8000)
test_images = load_mnist_images(DATA_DIR / "mnist/t10k-images-idx3-ubyte.gz", limit=2000)
test_labels = load_mnist_labels(DATA_DIR / "mnist/t10k-labels-idx1-ubyte.gz", limit=2000)
print("Sequence shape:", train_images.shape)
'''),
        code(r'''
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    train_images,
    train_labels,
    validation_split=0.15,
    epochs=3,
    batch_size=128,
    verbose=1,
)
'''),
        code(r'''
loss, acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test accuracy: {acc:.3f}")
pred = model.predict(test_images, verbose=0).argmax(axis=1)
print(classification_report(test_labels, pred))
ConfusionMatrixDisplay.from_predictions(test_labels, pred, cmap="Greys")
plt.title("LSTM Confusion Matrix")
plt.show()
'''),
    ]


def clip():
    return [
        md("""
        # CLIP-Style Contrastive Learning

        Full CLIP models need large pretrained weights and image-text datasets. This offline notebook
        demonstrates the core CLIP idea with a tiny PyTorch model: image embeddings are trained to align
        with text-label embeddings for handwritten digits.
        """),
        code(SETUP),
        code(r'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

digits = load_digits()
X = digits.images.astype("float32") / 16.0
y = digits.target.astype("int64")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train[:, None, :, :]), torch.tensor(y_train)),
    batch_size=128,
    shuffle=True,
)
test_x = torch.tensor(X_test[:, None, :, :], device=device)
test_y = torch.tensor(y_test, device=device)
prompts = [f"a handwritten digit {i}" for i in range(10)]
prompts
'''),
        code(r'''
class MiniCLIP(nn.Module):
    def __init__(self, embed_dim=32, num_classes=10):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, embed_dim),
        )
        self.text_encoder = nn.Embedding(num_classes, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, images):
        image_features = F.normalize(self.image_encoder(images), dim=-1)
        text_ids = torch.arange(10, device=images.device)
        text_features = F.normalize(self.text_encoder(text_ids), dim=-1)
        logits = self.logit_scale.exp() * image_features @ text_features.T
        return logits

model = MiniCLIP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(8):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    print(f"epoch {epoch + 1}: loss={total_loss / len(train_loader.dataset):.4f}")
'''),
        code(r'''
model.eval()
with torch.no_grad():
    logits = model(test_x)
    pred = logits.argmax(dim=1)
    accuracy = (pred == test_y).float().mean().item()
print(f"Zero-shot-style digit accuracy: {accuracy:.3f}")

fig, axes = plt.subplots(2, 5, figsize=(9, 4))
for ax, image, true_label, pred_label in zip(axes.ravel(), X_test[:10], y_test[:10], pred[:10].cpu().numpy()):
    ax.imshow(image, cmap="gray")
    ax.set_title(f"true {true_label} / {prompts[pred_label]}")
    ax.axis("off")
plt.tight_layout()
plt.show()
'''),
    ]


def ssd():
    return [
        md("""
        # Single Shot Detector

        This offline SSD notebook trains a tiny single-shot object detector on synthetic images containing
        one colored rectangle. The model predicts both class and bounding box in one forward pass.
        """),
        code(SETUP),
        code(r'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

CLASS_COLORS = np.array(
    [
        [1.0, 0.15, 0.15],
        [0.1, 0.7, 0.25],
        [0.15, 0.3, 1.0],
    ],
    dtype="float32",
)

def make_detection_data(n=640, size=64, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    images = np.zeros((n, 3, size, size), dtype="float32")
    boxes = np.zeros((n, 4), dtype="float32")
    labels = np.zeros(n, dtype="int64")
    for i in range(n):
        cls = rng.integers(0, len(CLASS_COLORS))
        w = rng.integers(size // 6, size // 3)
        h = rng.integers(size // 6, size // 3)
        x1 = rng.integers(2, size - w - 2)
        y1 = rng.integers(2, size - h - 2)
        x2 = x1 + w
        y2 = y1 + h
        images[i, :, y1:y2, x1:x2] = CLASS_COLORS[cls, :, None, None]
        images[i] += rng.normal(0, 0.02, images[i].shape).astype("float32")
        images[i] = np.clip(images[i], 0, 1)
        boxes[i] = [x1 / size, y1 / size, x2 / size, y2 / size]
        labels[i] = cls
    return images, boxes, labels

images, boxes, labels = make_detection_data()
train_images, test_images = images[:512], images[512:]
train_boxes, test_boxes = boxes[:512], boxes[512:]
train_labels, test_labels = labels[:512], labels[512:]

train_loader = DataLoader(
    TensorDataset(torch.tensor(train_images), torch.tensor(train_boxes), torch.tensor(train_labels)),
    batch_size=64,
    shuffle=True,
)
'''),
        code(r'''
class TinySSD(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.class_head = nn.Linear(64, num_classes)
        self.box_head = nn.Sequential(nn.Linear(64, 4), nn.Sigmoid())

    def forward(self, x):
        features = self.backbone(x)
        return self.class_head(features), self.box_head(features)

model = TinySSD().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    total_loss = 0
    for batch_images, batch_boxes, batch_labels in train_loader:
        batch_images = batch_images.to(device)
        batch_boxes = batch_boxes.to(device)
        batch_labels = batch_labels.to(device)
        class_logits, pred_boxes = model(batch_images)
        loss_cls = F.cross_entropy(class_logits, batch_labels)
        loss_box = F.smooth_l1_loss(pred_boxes, batch_boxes)
        loss = loss_cls + 5.0 * loss_box
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_labels)
    print(f"epoch {epoch + 1}: loss={total_loss / len(train_loader.dataset):.4f}")
'''),
        code(r'''
def draw_box(ax, box, color, label):
    size = 64
    x1, y1, x2, y2 = box * size
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
    ax.text(x1, max(0, y1 - 2), label, color=color, fontsize=9, weight="bold")

model.eval()
with torch.no_grad():
    test_tensor = torch.tensor(test_images, device=device)
    logits, pred_boxes = model(test_tensor)
    pred_labels = logits.argmax(dim=1).cpu().numpy()
    pred_boxes = pred_boxes.cpu().numpy()

class_acc = (pred_labels == test_labels).mean()
mean_box_error = np.abs(pred_boxes - test_boxes).mean()
print(f"Class accuracy: {class_acc:.3f}")
print(f"Mean absolute box error: {mean_box_error:.3f}")

fig, axes = plt.subplots(2, 4, figsize=(10, 5))
names = ["red box", "green box", "blue box"]
for ax, image, true_box, true_label, pred_box, pred_label in zip(
    axes.ravel(), test_images[:8], test_boxes[:8], test_labels[:8], pred_boxes[:8], pred_labels[:8]
):
    ax.imshow(np.moveaxis(image, 0, -1))
    draw_box(ax, true_box, "white", f"true {names[true_label]}")
    draw_box(ax, pred_box, "black", f"pred {names[pred_label]}")
    ax.axis("off")
plt.tight_layout()
plt.show()
'''),
    ]


def naive_bayes():
    return [
        md("""
        # Naive Bayes

        Use the course purchase sample to predict whether a customer buys based on day type,
        discount, and free-delivery flags.
        """),
        code(SETUP),
        code(r'''
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv(DATA_DIR / "naive_bayes_purchase.csv")
for col in df.columns:
    df[col] = df[col].astype(str).str.strip()
display(df.head())
display(df["Purchase"].value_counts().rename_axis("Purchase").reset_index(name="count"))
'''),
        code(r'''
X = df.drop(columns=["Purchase"])
y = df["Purchase"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

model = Pipeline(
    steps=[
        ("encode", OneHotEncoder(handle_unknown="ignore")),
        ("nb", MultinomialNB(alpha=1.0)),
    ]
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred))
ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Greys")
plt.title("Naive Bayes Confusion Matrix")
plt.show()

result = X_test.copy()
result["actual"] = y_test.values
result["predicted"] = pred
for idx, cls in enumerate(model.named_steps["nb"].classes_):
    result[f"prob_{cls}"] = proba[:, idx].round(3)
display(result)
'''),
        code(r'''
feature_names = model.named_steps["encode"].get_feature_names_out(X.columns)
log_probs = pd.DataFrame(
    model.named_steps["nb"].feature_log_prob_,
    index=model.named_steps["nb"].classes_,
    columns=feature_names,
).T
display(log_probs.sort_values(model.named_steps["nb"].classes_[0], ascending=False))
'''),
    ]


NOTEBOOKS = {
    "01_linear_regression.ipynb": linear_regression,
    "02_logistic_regression.ipynb": logistic_regression,
    "03_kmeans_clustering.ipynb": kmeans_clustering,
    "04_knn.ipynb": knn,
    "05_svm.ipynb": svm,
    "06_decision_tree.ipynb": decision_tree,
    "07_random_forest.ipynb": random_forest,
    "08_xgboost.ipynb": xgboost,
    "09_cnn.ipynb": cnn,
    "10_ann.ipynb": ann,
    "11_lstm.ipynb": lstm,
    "12_clip.ipynb": clip,
    "13_ssd.ipynb": ssd,
    "14_naive_bayes.ipynb": naive_bayes,
}


README = """# MLDL Algorithm Notebooks

This folder contains runnable notebooks for the requested machine-learning and deep-learning algorithms.

## Notebooks

1. Linear Regression - company profit prediction with the course `1000_Companies.csv`.
2. Logistic Regression - breast cancer diagnosis with the course logistic-regression CSV.
3. K-Means Clustering - car clustering with the course cars CSV.
4. KNN - diabetes classification with the course diabetes CSV.
5. SVM - SVM margin demo plus breast cancer classification.
6. Decision Tree - course decision-tree dataset.
7. Random Forest - iris classification using the course iris CSV.
8. XGBoost - red wine quality classification; falls back to scikit-learn boosting if `xgboost` is unavailable.
9. CNN - MNIST digit recognition using the course MNIST gzip files.
10. ANN - dense neural network on local MNIST.
11. LSTM - row-sequence MNIST classifier.
12. CLIP - offline CLIP-style contrastive image/text alignment on digit prompts.
13. SSD - offline single-shot detector on synthetic rectangle images.
14. Naive Bayes - purchase prediction using the course Naive Bayes sample converted to CSV.

## Data

Datasets from the uploaded course zip are stored in `MLDL/data`. The CLIP and SSD notebooks are self-contained
because the local environment does not include pretrained CLIP/SSD packages or internet downloads.

Run notebooks from the `MLDL/notebooks` directory or from the repository root; each notebook resolves the data path both ways.
"""


def main():
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    convert_naive_bayes_xlsx()

    for filename, builder in NOTEBOOKS.items():
        write_notebook(NOTEBOOK_DIR / filename, builder())

    (ROOT / "README.md").write_text(README, encoding="utf-8")
    print(f"Wrote {len(NOTEBOOKS)} notebooks to {NOTEBOOK_DIR}")


if __name__ == "__main__":
    main()
