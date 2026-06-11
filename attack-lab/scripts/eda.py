"""Exploratory data analysis for the cleaned network attack dataset."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = PROJECT_ROOT / "csv"
RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_PATH = CSV_DIR / "final_dataset_clean.csv"

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "attack-lab-matplotlib"),
)

import matplotlib.pyplot as plt
import seaborn as sns

LABEL_NAMES = {
    0: "Normal",
    1: "Ping",
    2: "SYN",
}


def save_current_plot(filename: str) -> None:
    output_path = RESULTS_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot: {output_path.relative_to(PROJECT_ROOT)}")


def print_basic_info(df: pd.DataFrame) -> None:
    print("\n=== Basic Info ===")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(list(df.columns))
    print("\nData types:")
    print(df.dtypes)


def print_summary_statistics(df: pd.DataFrame) -> None:
    print("\n=== Summary Statistics ===")
    print(df.describe(include="all"))


def print_missing_values(df: pd.DataFrame) -> None:
    print("\n=== Missing Values ===")
    print(df.isna().sum())


def print_class_distribution(df: pd.DataFrame) -> pd.Series:
    print("\n=== Class Distribution ===")
    distribution = (
        df["label"]
        .map(LABEL_NAMES)
        .fillna("Unknown")
        .value_counts()
        .reindex(["Normal", "Ping", "SYN"], fill_value=0)
    )
    print(distribution)
    return distribution


def plot_class_distribution(distribution: pd.Series) -> None:
    plot_df = distribution.rename_axis("label_name").reset_index(name="count")
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=plot_df,
        x="label_name",
        y="count",
        hue="label_name",
        palette="Set2",
        legend=False,
    )
    plt.title("Class Distribution")
    plt.xlabel("Traffic Class")
    plt.ylabel("Packet Count")
    save_current_plot("class_distribution.png")


def plot_packet_length_histogram(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="frame.len", hue="label_name", bins=40, kde=True)
    plt.title("Packet Length Distribution")
    plt.xlabel("Packet Length")
    plt.ylabel("Count")
    save_current_plot("packet_length_histogram.png")


def plot_packet_length_boxplot(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="label_name",
        y="frame.len",
        hue="label_name",
        palette="Set2",
        legend=False,
    )
    plt.title("Packet Length vs Label")
    plt.xlabel("Traffic Class")
    plt.ylabel("Packet Length")
    save_current_plot("packet_length_vs_label_boxplot.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        print("Skipping correlation heatmap: no numeric columns found.")
        return

    plt.figure(figsize=(12, 9))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    save_current_plot("correlation_heatmap.png")


def print_observations(df: pd.DataFrame) -> None:
    print("\n=== EDA Observations ===")
    print("# Attack and normal traffic often differ in packet size, protocol,")
    print("# TCP flag patterns, ICMP fields, TTL values, and packet timing.")
    print("# Ping flood traffic is expected to show strong ICMP activity through")
    print("# icmp.type and icmp.code, often with repeated packet lengths.")
    print("# SYN-like traffic is expected to show higher tcp.flags.syn values and")
    print("# TCP port activity compared with normal traffic.")
    print("# Important candidate features for ML include frame.len, ip.proto,")
    print("# tcp.flags.syn, tcp.flags.ack, tcp.flags.reset, icmp.type, icmp.code,")
    print("# ip.ttl, tcp.srcport, tcp.dstport, and inter_arrival.")

    grouped = df.groupby("label_name", dropna=False).mean(numeric_only=True)
    if not grouped.empty:
        print("\nMean numeric values by class:")
        print(grouped)


def main() -> None:
    if not DATASET_PATH.exists() or DATASET_PATH.stat().st_size == 0:
        raise SystemExit(
            "Clean dataset not found. Run scripts/pcap_to_csv.py and "
            "scripts/preprocess.py before EDA."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(DATASET_PATH)
    if "label_name" not in df.columns:
        df["label_name"] = df["label"].map(LABEL_NAMES).fillna("Unknown")

    print_basic_info(df)
    print_summary_statistics(df)
    print_missing_values(df)
    distribution = print_class_distribution(df)

    plot_class_distribution(distribution)
    plot_packet_length_histogram(df)
    plot_packet_length_boxplot(df)
    plot_correlation_heatmap(df)
    print_observations(df)


if __name__ == "__main__":
    main()
