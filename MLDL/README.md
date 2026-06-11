# MLDL Algorithm Notebooks

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
