"""Convert Wireshark PCAPNG captures into CSV feature files.

This script requires `tshark` to be installed and available on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PCAP_DIR = PROJECT_ROOT / "pcaps"
CSV_DIR = PROJECT_ROOT / "csv"

FIELDS = [
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

CAPTURES = {
    "normal_traffic.pcapng": "normal.csv",
    "ping_flood.pcapng": "ping.csv",
    "syn_like_traffic.pcapng": "syn.csv",
}


def build_tshark_command(input_file: Path) -> list[str]:
    """Build the tshark command for extracting packet fields."""
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
        "quote=d",
        "-E",
        "occurrence=f",
    ]

    for field in FIELDS:
        command.extend(["-e", field])

    return command


def convert_capture(input_name: str, output_name: str) -> None:
    input_file = PCAP_DIR / input_name
    output_file = CSV_DIR / output_name

    if not input_file.exists():
        raise FileNotFoundError(f"Missing capture file: {input_file}")

    CSV_DIR.mkdir(parents=True, exist_ok=True)

    command = build_tshark_command(input_file)
    with output_file.open("w", encoding="utf-8", newline="") as csv_handle:
        result = subprocess.run(
            command,
            stdout=csv_handle,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"tshark failed for {input_file.name} with exit code "
            f"{result.returncode}:\n{result.stderr.strip()}"
        )

    print(f"Converted {input_file.name} -> {output_file.relative_to(PROJECT_ROOT)}")


def main() -> None:
    if shutil.which("tshark") is None:
        raise SystemExit(
            "tshark was not found on PATH. Install Wireshark/tshark before "
            "running this script."
        )

    for input_name, output_name in CAPTURES.items():
        convert_capture(input_name, output_name)


if __name__ == "__main__":
    main()
