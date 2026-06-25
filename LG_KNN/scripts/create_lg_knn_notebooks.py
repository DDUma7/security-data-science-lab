from pathlib import Path

import nbformat as nbf


OUTPUT_DIR = Path("LG_KNN/notebooks")


COMMON_IMPORTS = r'''
import os
from collections import Counter
from pathlib import Path
import re
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from wordcloud import WordCloud

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
BW_COLORS = ["#111111", "#666666", "#999999", "#D9D9D9"]
BW_HATCHES = ["///", "\\\\\\", "xx", "..", "++"]
BW_LINESTYLES = ["-", "--", "-.", ":"]
BW_MARKERS = ["o", "s", "^", "D"]

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
})
pd.set_option("display.max_colwidth", 160)
'''.strip()


HELPERS = r'''
def basic_text_eda(data, label_col="label", text_col="text"):
    """Create reusable text statistics for EDA."""
    eda_df = data.copy()
    eda_df[text_col] = eda_df[text_col].fillna("").astype(str)
    eda_df["char_count"] = eda_df[text_col].str.len()
    eda_df["word_count"] = eda_df[text_col].str.split().str.len()
    eda_df["unique_word_count"] = eda_df[text_col].str.lower().str.findall(r"[a-zA-Z]{2,}").map(lambda words: len(set(words)))
    eda_df["avg_word_length"] = eda_df[text_col].str.findall(r"[a-zA-Z]+").map(
        lambda words: np.mean([len(word) for word in words]) if words else 0
    )
    return eda_df


def add_bar_labels(ax):
    for patch in ax.patches:
        height = patch.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            f"{height:,.0f}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )


def show_label_distribution(data, label_col="label"):
    label_counts = data[label_col].value_counts().rename_axis(label_col).reset_index(name="count")
    label_counts["percentage"] = (label_counts["count"] / len(data) * 100).round(2)
    display(label_counts)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        label_counts[label_col],
        label_counts["count"],
        color=["white", "#E6E6E6", "#BFBFBF", "#8C8C8C"][: len(label_counts)],
        edgecolor="black",
        linewidth=1.2,
    )
    for idx, bar in enumerate(bars):
        bar.set_hatch(BW_HATCHES[idx % len(BW_HATCHES)])
    ax.set_title("Class Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Number of records")
    ax.tick_params(axis="x", rotation=20)
    add_bar_labels(ax)
    ax.margins(y=0.15)
    plt.tight_layout()
    plt.show()


def plot_text_lengths(eda_df, label_col="label"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    labels = sorted(eda_df[label_col].dropna().unique())
    upper_chars = eda_df["char_count"].quantile(0.95)
    bins = np.linspace(0, upper_chars, 35)

    for idx, label_value in enumerate(labels):
        clipped_chars = eda_df.loc[eda_df[label_col] == label_value, "char_count"].clip(upper=upper_chars)
        axes[0].hist(
            clipped_chars,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=BW_COLORS[idx % len(BW_COLORS)],
            linestyle=BW_LINESTYLES[idx % len(BW_LINESTYLES)],
            label=str(label_value),
        )
    axes[0].set_title("Character Count Distribution (95th percentile clipped)")
    axes[0].set_xlabel("Characters")
    axes[0].set_ylabel("Density")
    axes[0].legend(title="Class")

    upper_words = eda_df["word_count"].quantile(0.95)
    box_data = [
        eda_df.loc[eda_df[label_col] == label_value, "word_count"].clip(upper=upper_words)
        for label_value in labels
    ]
    box = axes[1].boxplot(
        box_data,
        tick_labels=labels,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        boxprops={"edgecolor": "black", "linewidth": 1.2},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 3,
        },
    )
    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(["white", "#E6E6E6", "#BFBFBF", "#8C8C8C"][idx % 4])
        patch.set_hatch(BW_HATCHES[idx % len(BW_HATCHES)])
    axes[1].set_title("Word Count by Class (95th percentile clipped)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Words")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.show()


def top_words_by_label(data, label_value, text_col="text", label_col="label", n=20):
    text = " ".join(data.loc[data[label_col] == label_value, text_col].dropna().astype(str).str.lower())
    words = re.findall(r"[a-zA-Z]{3,}", text)
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return pd.DataFrame(Counter(words).most_common(n), columns=["word", "count"])


def wordcloud_text(data, label_value, text_col="text", label_col="label"):
    text = " ".join(data.loc[data[label_col] == label_value, text_col].dropna().astype(str))
    return text.strip() if text.strip() else "empty"


def plot_word_clouds(data, text_col="text", label_col="label"):
    labels = sorted(data[label_col].dropna().unique())
    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 5))
    if len(labels) == 1:
        axes = [axes]

    stopwords = set(ENGLISH_STOP_WORDS)
    for idx, label_value in enumerate(labels):
        wc = WordCloud(
            width=900,
            height=500,
            background_color="white",
            stopwords=stopwords,
            max_words=120,
            collocations=False,
            random_state=RANDOM_STATE,
            color_func=lambda *args, **kwargs: BW_COLORS[idx % 3],
        ).generate(wordcloud_text(data, label_value, text_col=text_col, label_col=label_col))
        axes[idx].imshow(wc, interpolation="bilinear")
        axes[idx].set_title(f"Word Cloud: {label_value}")
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


def plot_top_words(words_df, title, n=15):
    plot_df = words_df.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        plot_df["word"],
        plot_df["count"],
        color="white",
        edgecolor="black",
        linewidth=1.0,
    )
    for idx, bar in enumerate(bars):
        bar.set_hatch(BW_HATCHES[idx % len(BW_HATCHES)])
    ax.set_title(title)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_df):
    metrics_to_plot = ["accuracy", "weighted_f1", "roc_auc"]
    plot_df = results_df.set_index("model")[metrics_to_plot]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(plot_df.index))
    width = 0.24

    for idx, metric in enumerate(metrics_to_plot):
        values = plot_df[metric].values
        bars = ax.bar(
            x + (idx - 1) * width,
            values,
            width=width,
            label=metric,
            color=["white", "#E6E6E6", "#BFBFBF"][idx],
            edgecolor="black",
            linewidth=1.0,
        )
        for bar in bars:
            bar.set_hatch(BW_HATCHES[idx % len(BW_HATCHES)])

    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=15, ha="right")
    ax.legend(title="Metric")
    plt.tight_layout()
    plt.show()


def evaluate_model(name, model, X_train, X_test, y_train, y_test, positive_label):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
    }

    print(f"{name} classification report")
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).T
    display(report)

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Greys", values_format="d", xticks_rotation=30, colorbar=False)
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.show()

    if len(labels) == 2 and hasattr(model, "predict_proba"):
        try:
            positive_index = list(model.classes_).index(positive_label)
            y_score = model.predict_proba(X_test)[:, positive_index]
            y_test_binary = (y_test == positive_label).astype(int)
            metrics["roc_auc"] = roc_auc_score(y_test_binary, y_score)
            RocCurveDisplay.from_predictions(
                y_test_binary,
                y_score,
                name=name,
                color="black",
                linestyle="-" if name == "Logistic Regression" else "--",
                marker="o" if name == "Logistic Regression" else "s",
                markevery=0.1,
                linewidth=1.8,
            )
            plt.title(f"{name} ROC Curve")
            plt.tight_layout()
            plt.show()
        except ValueError:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan

    return metrics, model
'''.strip()


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(source):
    return nbf.v4.new_code_cell(source.strip())


def build_notebook(config):
    title = config["title"]
    data_path = config["data_path"]
    output_name = config["output_name"]
    positive_label = config["positive_label"]
    loader_code = config["loader_code"].format(data_path=data_path)

    cells = [
        markdown(
            f"""
# {title}

This notebook performs exploratory data analysis and builds two supervised text classifiers:

- Logistic Regression with TF-IDF features
- K-Nearest Neighbors with TF-IDF features

Dataset path: `{data_path}`

Plotting note: charts use grayscale, hatches, line styles, and markers so they remain readable in black-and-white xerox copies.
"""
        ),
        markdown("## 1. Imports and Settings"),
        code(COMMON_IMPORTS),
        markdown("## 1.1 Helper Functions"),
        code(HELPERS),
        markdown("## 2. Load Dataset"),
        code(loader_code),
        markdown("## 3. Basic Structure"),
        code(
            r'''
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
display(df.head())

print("\nData types and missing values")
display(pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing": df.isna().sum(),
    "missing_pct": (df.isna().mean() * 100).round(2),
}))
'''
        ),
        markdown("## 4. Class Balance"),
        code("show_label_distribution(df, label_col=\"label\")"),
        markdown("## 5. Missing Values, Blank Text, and Duplicates"),
        code(
            r'''
quality_summary = pd.DataFrame({
    "metric": [
        "rows",
        "missing_text",
        "blank_text",
        "missing_label",
        "duplicate_rows",
        "duplicate_texts",
    ],
    "value": [
        len(df),
        df["text"].isna().sum(),
        df["text"].fillna("").str.strip().eq("").sum(),
        df["label"].isna().sum(),
        df.duplicated().sum(),
        df["text"].duplicated().sum(),
    ],
})
display(quality_summary)

display(df[df["text"].duplicated(keep=False)].sort_values("text").head(10))
'''
        ),
        markdown("## 6. Text Length Analysis"),
        code(
            r'''
eda_df = basic_text_eda(df)

display(
    eda_df.groupby("label")[["char_count", "word_count", "unique_word_count", "avg_word_length"]]
    .describe()
    .round(2)
)

plot_text_lengths(eda_df)
'''
        ),
        markdown("## 7. Word Cloud Visualization"),
        code(
            r'''
plot_word_clouds(df, text_col="text", label_col="label")
'''
        ),
        markdown("## 8. Most Frequent Words by Class"),
        code(
            r'''
for label_value in sorted(df["label"].dropna().unique()):
    print(f"Top words for: {label_value}")
    words_df = top_words_by_label(df, label_value, n=20)
    display(words_df)
    plot_top_words(words_df, title=f"Top Words: {label_value}", n=15)
'''
        ),
        markdown(
            """
## 9. Text Representation Method

The text representation used for both models is **TF-IDF** using `TfidfVectorizer`.

TF-IDF means Term Frequency-Inverse Document Frequency. It converts text into numerical features by giving higher weight to words that are important in a document but not too common across the full dataset. This is a sparse vector representation, not a dense neural word embedding such as Word2Vec, GloVe, FastText, or BERT.
"""
        ),
        markdown("## 10. Prepare Data for Modeling"),
        code(
            rf'''
POSITIVE_LABEL = {positive_label!r}

model_df = df.dropna(subset=["label"]).copy()
model_df["text"] = model_df["text"].fillna("").astype(str)
model_df = model_df[model_df["text"].str.strip().ne("")]
model_df = model_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print("Rows after cleaning:", len(model_df))
show_label_distribution(model_df, label_col="label")

X = model_df["text"]
y = model_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])
'''
        ),
        markdown("## 11. Logistic Regression"),
        code(
            r'''
logistic_regression = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_features=12000,
            ),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

results = []
trained_models = {}

metrics, trained_model = evaluate_model(
    "Logistic Regression",
    logistic_regression,
    X_train,
    X_test,
    y_train,
    y_test,
    positive_label=POSITIVE_LABEL,
)
results.append(metrics)
trained_models["Logistic Regression"] = trained_model
'''
        ),
        markdown("## 12. K-Nearest Neighbors"),
        code(
            r'''
knn = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_features=7000,
            ),
        ),
        (
            "model",
            KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                metric="cosine",
                algorithm="brute",
                n_jobs=-1,
            ),
        ),
    ]
)

metrics, trained_model = evaluate_model(
    "KNN",
    knn,
    X_train,
    X_test,
    y_train,
    y_test,
    positive_label=POSITIVE_LABEL,
)
results.append(metrics)
trained_models["KNN"] = trained_model
'''
        ),
        markdown("## 13. Model Comparison"),
        code(
            r'''
results_df = pd.DataFrame(results).sort_values("weighted_f1", ascending=False).reset_index(drop=True)
display(results_df.round(4))
plot_model_comparison(results_df)

best_model_name = results_df.loc[0, "model"]
print(f"Best model by weighted F1: {best_model_name}")
'''
        ),
        markdown("## 14. Quick Error Check"),
        code(
            r'''
best_model = trained_models[best_model_name]
y_pred = best_model.predict(X_test)

errors = pd.DataFrame({
    "text": X_test,
    "actual": y_test,
    "predicted": y_pred,
})
errors = errors[errors["actual"] != errors["predicted"]]

print("Number of misclassified examples:", len(errors))
display(errors.head(15))
'''
        ),
        markdown(
            """
## Conclusion

Use the comparison table above to report which model performed better on this train/test split. In most text-classification problems, Logistic Regression with TF-IDF is a strong baseline because it handles sparse high-dimensional text features well. KNN is included for comparison, but it can be slower and more sensitive to feature dimensionality.
"""
        ),
    ]

    nb = nbf.v4.new_notebook(cells=cells)
    nb["metadata"] = {
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

    output_path = OUTPUT_DIR / output_name
    nbf.write(nb, output_path)
    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = [
        {
            "title": "Phishing Emails EDA, Logistic Regression, and KNN",
            "data_path": "/home/durgaumadev/Documents/MTech/Security for Data Science/phishing/Phishing_Emails.csv",
            "output_name": "phishing_emails_eda_logistic_knn.ipynb",
            "positive_label": "Phishing Email",
            "loader_code": r'''
DATA_PATH = Path(r"{data_path}")
df_raw = pd.read_csv(DATA_PATH)

df = (
    df_raw[["Email Text", "Email Type"]]
    .rename(columns={{"Email Text": "text", "Email Type": "label"}})
    .copy()
)

df["text"] = df["text"].fillna("").astype(str)
df["label"] = df["label"].astype(str)
'''
        },
        {
            "title": "SMS Spam EDA, Logistic Regression, and KNN",
            "data_path": "/home/durgaumadev/Documents/MTech/Security for Data Science/sms+spam+collection/SMSSpamCollection",
            "output_name": "sms_spam_eda_logistic_knn.ipynb",
            "positive_label": "spam",
            "loader_code": r'''
DATA_PATH = Path(r"{data_path}")
df = pd.read_csv(DATA_PATH, sep="\t", names=["label", "text"], header=None)

df["text"] = df["text"].fillna("").astype(str)
df["label"] = df["label"].astype(str)
'''
        },
    ]

    for config in configs:
        output_path = build_notebook(config)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
