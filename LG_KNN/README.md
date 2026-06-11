# LG_KNN

This folder contains the Logistic Regression and KNN text-classification notebooks for:

- Phishing email detection
- SMS spam detection

## Files

- `notebooks/phishing_emails_eda_logistic_knn.ipynb`
- `notebooks/sms_spam_eda_logistic_knn.ipynb`
- `scripts/create_lg_knn_notebooks.py`

## Text Representation

The notebooks use **TF-IDF vectorization** with `TfidfVectorizer`.

This is a sparse text feature representation. It is not a dense neural word embedding method like Word2Vec, GloVe, FastText, or BERT.

## Plot Style

All notebook plots use black-and-white friendly design:

- Grayscale colors
- Hatch patterns for bars
- Line styles and markers for curves
- Grayscale confusion matrices
- Grayscale word-cloud visualizations

This is intended to keep plots readable after xerox/photocopy printing.

## Run Again

From the repository root:

```bash
.venv/bin/python LG_KNN/scripts/create_lg_knn_notebooks.py
MPLCONFIGDIR=/tmp .venv/bin/jupyter nbconvert --to notebook --execute --inplace LG_KNN/notebooks/sms_spam_eda_logistic_knn.ipynb
MPLCONFIGDIR=/tmp .venv/bin/jupyter nbconvert --to notebook --execute --inplace LG_KNN/notebooks/phishing_emails_eda_logistic_knn.ipynb
```

The notebooks load the datasets from the original paths provided for this project.
