# attack-lab

## Project Overview

`attack-lab` is a machine learning project for detecting network attack traffic from packet captures. The project is organized to collect Wireshark `.pcapng` files, extract packet-level features into CSV datasets, preprocess the data, train a classifier, and evaluate results.

## Dataset Creation

The dataset is built from both normal and attack traffic:

- Normal traffic is captured during regular network activity and saved as `pcaps/normal_traffic.pcapng`.
- Ping flood traffic is captured as an ICMP-based attack sample and saved as `pcaps/ping_flood.pcapng`.
- SYN-like traffic is captured as a TCP-based attack sample and saved as `pcaps/syn_like_traffic.pcapng`.
- New unseen test traffic can be captured and saved as `pcaps/new_test_capture.pcapng`.

Extracted CSV files are stored in `csv/`. Raw extracted data can be combined into `final_dataset_raw.csv`, then cleaned and prepared for modeling as `final_dataset_clean.csv`.

## Machine Learning Model

The primary model used in this project is a Random Forest classifier. Random Forest is a strong baseline for network attack detection because it handles mixed feature types well, is robust to noisy data, and provides reliable performance on tabular datasets.

## Workflow Steps

1. Capture normal and attack traffic using Wireshark.
2. Save capture files in the `pcaps/` folder.
3. Convert packet captures to CSV files using `scripts/pcap_to_csv.py`.
4. Clean, label, and combine datasets using `scripts/preprocess.py`.
5. Train the Random Forest model using `scripts/train_model.py`.
6. Test the model on new capture data using `scripts/test_new_data.py`.
7. Save evaluation outputs, graphs, and reports in the `results/` folder.

## Project Structure

```text
attack-lab/
├── csv/
│   ├── final_dataset_clean.csv
│   ├── final_dataset_raw.csv
│   ├── normal.csv
│   ├── ping.csv
│   └── syn.csv
├── notebooks/
│   └── attack_detection.ipynb
├── pcaps/
│   ├── new_test_capture.pcapng
│   ├── normal_traffic.pcapng
│   ├── ping_flood.pcapng
│   └── syn_like_traffic.pcapng
├── results/
│   ├── classification_report.txt
│   └── confusion_matrix.png
├── scripts/
│   ├── pcap_to_csv.py
│   ├── preprocess.py
│   ├── test_new_data.py
│   └── train_model.py
└── README.md
```
