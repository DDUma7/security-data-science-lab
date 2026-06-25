"""Test a trained attack-detection model on a new PCAPNG capture.

This script converts ``pcaps/new_test_capture.pcapng`` to CSV with tshark,
preprocesses the packet-level features, loads the trained Random Forest model,
predicts traffic labels, and saves the prediction output.

Run from the attack-lab directory or repository root with:

    python3 scripts/test_new_data.py
"""

from __future__ import annotations

import shutil
import subprocess
import os
from pathlib import Path

import joblib

# Keep Matplotlib from trying to write inside the user's home config directory.
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib
import pandas as pd


# Use a non-interactive backend so the script is safe to run in terminals/CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PCAP_PATH = PROJECT_ROOT / "pcaps" / "new_test_capture.pcapng"
CSV_DIR = PROJECT_ROOT / "csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_attack_detection.joblib"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_TEST_CSV = CSV_DIR / "new_test.csv"
PREDICTIONS_CSV = CSV_DIR / "new_test_predictions.csv"
PREDICTION_PLOT = RESULTS_DIR / "new_test_prediction_distribution.png"

TSHARK_FIELDS = [
    "frame.time_epoch",
    "ip.src",
    "ip.dst",
    "ip.proto",
    "frame.len",
    "tcp.srcport",
    "tcp.dstport",
    "tcp.flags.syn",
    "tcp.flags.ack",
    "tcp.flags.reset",
    "ip.ttl",
    "icmp.type",
    "icmp.code",
]

NUMERIC_COLUMNS = [
    "frame.time_epoch",
    "ip.proto",
    "frame.len",
    "tcp.srcport",
    "tcp.dstport",
    "tcp.flags.syn",
    "tcp.flags.ack",
    "tcp.flags.reset",
    "ip.ttl",
    "icmp.type",
    "icmp.code",
]

TRAINING_FEATURES = [
    "ip.proto",
    "frame.len",
    "tcp.srcport",
    "tcp.dstport",
    "tcp.flags.syn",
    "tcp.flags.ack",
    "tcp.flags.reset",
    "ip.ttl",
    "icmp.type",
    "icmp.code",
    "inter_arrival",
]

DEFAULT_LABEL_MAP = {
    0: "Normal",
    1: "Ping_Flood",
    2: "SYN_Flood",
}


def build_tshark_command(input_file: Path) -> list[str]:
    """Build the tshark command used to extract packet fields."""
    command = [
        "tshark",
        "-r",
        str(input_file),
        "-T",
        "fields",
        "-E",
        "header=y",
        "-E",
        "separator=,",
        "-E",
        "occurrence=f",
    ]

    for field in TSHARK_FIELDS:
        command.extend(["-e", field])

    return command


def convert_pcap_to_csv() -> None:
    """Convert the new PCAPNG capture to csv/new_test.csv."""
    if shutil.which("tshark") is None:
        raise SystemExit(
            "tshark was not found on PATH. Install Wireshark/tshark before "
            "running this script."
        )

    if not PCAP_PATH.exists():
        raise FileNotFoundError(f"Missing PCAP file: {PCAP_PATH}")

    CSV_DIR.mkdir(parents=True, exist_ok=True)

    command = build_tshark_command(PCAP_PATH)
    with RAW_TEST_CSV.open("w", encoding="utf-8", newline="") as csv_handle:
        result = subprocess.run(
            command,
            stdout=csv_handle,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"tshark failed with exit code {result.returncode}:\n"
            f"{result.stderr.strip()}"
        )

    print(f"Converted PCAP to CSV: {RAW_TEST_CSV.relative_to(PROJECT_ROOT)}")


def load_and_preprocess_data(feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load csv/new_test.csv and prepare model features safely."""
    if not RAW_TEST_CSV.exists() or RAW_TEST_CSV.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty CSV file: {RAW_TEST_CSV}")

    df = pd.read_csv(RAW_TEST_CSV)

    # Ensure all expected tshark columns exist, even if a capture omits them.
    for column in TSHARK_FIELDS:
        if column not in df.columns:
            df[column] = 0

    # Convert numeric packet fields and fill missing values.
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["ip.src"] = df["ip.src"].fillna("0")
    df["ip.dst"] = df["ip.dst"].fillna("0")
    df = df.fillna(0)

    # Sort by packet time and create the inter-arrival timing feature.
    df = df.sort_values("frame.time_epoch").reset_index(drop=True)
    df["inter_arrival"] = df["frame.time_epoch"].diff().fillna(0)

    # Make sure every training feature exists before prediction.
    for column in feature_columns:
        if column not in df.columns:
            df[column] = 0

    X_new = df[feature_columns].copy()
    X_new = X_new.apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"Loaded new test data with shape: {df.shape}")
    print(f"Using model features: {feature_columns}")
    return df, X_new


def load_model_artifact() -> tuple[object, list[str], dict[int, str]]:
    """Load the saved Random Forest joblib artifact."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing trained model file: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_columns = artifact.get("feature_columns", TRAINING_FEATURES)
    label_map = artifact.get("label_map", DEFAULT_LABEL_MAP)

    # Joblib preserves integer keys, but normalize defensively for safety.
    label_map = {int(key): value for key, value in label_map.items()}

    print(f"Loaded model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    return model, feature_columns, label_map


def predict_labels(
    df: pd.DataFrame,
    X_new: pd.DataFrame,
    model: object,
    label_map: dict[int, str],
) -> pd.DataFrame:
    """Predict traffic classes and append readable labels to the dataframe."""
    predictions = model.predict(X_new)
    predicted_labels = pd.Series(predictions).astype(int).map(label_map).fillna("Unknown")

    df = df.copy()
    df["predicted_label"] = predicted_labels

    prediction_counts = df["predicted_label"].value_counts()
    print("\nPrediction counts:")
    print(prediction_counts)

    df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\nSaved predictions to: {PREDICTIONS_CSV.relative_to(PROJECT_ROOT)}")

    return df


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Plot and save the predicted class distribution."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    order = ["Normal", "Ping_Flood", "SYN_Flood"]
    counts = df["predicted_label"].value_counts().reindex(order, fill_value=0)

    plt.figure(figsize=(8, 5))
    if sns is not None:
        sns.countplot(
            data=df,
            x="predicted_label",
            hue="predicted_label",
            order=order,
            palette="Set2",
            legend=False,
        )
    else:
        print("Warning: seaborn is not installed; using matplotlib for the plot.")
        plt.bar(counts.index, counts.values, color=["#66c2a5", "#fc8d62", "#8da0cb"])

    plt.title("New Test Capture Prediction Distribution")
    plt.xlabel("Predicted Traffic Class")
    plt.ylabel("Packet Count")
    plt.tight_layout()
    plt.savefig(PREDICTION_PLOT, dpi=150)
    plt.close()

    print(f"Saved class distribution plot to: {PREDICTION_PLOT.relative_to(PROJECT_ROOT)}")


def print_final_interpretation(df: pd.DataFrame) -> None:
    """Print a short majority-class interpretation."""
    if df.empty:
        print("\nNo packets were available for prediction.")
        return

    majority_label = df["predicted_label"].value_counts().idxmax()
    print(
        "\nThe traffic is predominantly "
        f"{majority_label} based on majority prediction"
    )


def main() -> None:
    """Run the full PCAP conversion, prediction, and reporting workflow."""
    convert_pcap_to_csv()
    model, feature_columns, label_map = load_model_artifact()
    df, X_new = load_and_preprocess_data(feature_columns)
    prediction_df = predict_labels(df, X_new, model, label_map)
    plot_class_distribution(prediction_df)
    print_final_interpretation(prediction_df)


if __name__ == "__main__":
    main()
