"""Merge extracted packet CSVs and create a clean labeled dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = PROJECT_ROOT / "csv"

EXPECTED_COLUMNS = [
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

DATASETS = [
    ("normal.csv", 0, "Normal"),
    ("ping.csv", 1, "Ping"),
    ("syn.csv", 2, "SYN"),
]


def load_packet_csv(filename: str, label: int, label_name: str) -> pd.DataFrame:
    csv_path = CSV_DIR / filename

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print(f"Warning: {csv_path.name} is missing or empty; skipping rows.")
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    else:
        df = pd.read_csv(csv_path)

    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            df[column] = 0

    df = df[EXPECTED_COLUMNS].copy()
    df["label"] = label
    df["label_name"] = label_name
    df["source_file"] = filename
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["ip.src"] = df["ip.src"].fillna("0")
    df["ip.dst"] = df["ip.dst"].fillna("0")
    df["label_name"] = df["label_name"].fillna("Unknown")
    df["source_file"] = df["source_file"].fillna("unknown")

    df = df.fillna(0)
    df = df.sort_values("frame.time_epoch").reset_index(drop=True)
    df["inter_arrival"] = df["frame.time_epoch"].diff().fillna(0)

    return df


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    frames = [load_packet_csv(*dataset) for dataset in DATASETS]
    merged = pd.concat(frames, ignore_index=True)

    raw_path = CSV_DIR / "final_dataset_raw.csv"
    clean_path = CSV_DIR / "final_dataset_clean.csv"

    merged.to_csv(raw_path, index=False)

    clean = clean_dataset(merged)
    clean.to_csv(clean_path, index=False)

    print(f"Saved raw dataset: {raw_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved clean dataset: {clean_path.relative_to(PROJECT_ROOT)}")
    print(f"Final shape: {clean.shape}")


if __name__ == "__main__":
    main()
