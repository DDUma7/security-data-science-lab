# SVM & DT

This folder contains the Support Vector Machine and Decision Tree text-classification notebooks for:

- Phishing email detection
- SMS spam detection

## Files

- `notebooks/phishing_emails_eda_svm_dt.ipynb`
- `notebooks/sms_spam_eda_svm_dt.ipynb`
- `scripts/create_svm_dt_notebooks.py`

## Text Representation

The notebooks use **TF-IDF vectorization** with `TfidfVectorizer`.

This is a sparse text feature representation. It is not a dense neural word embedding method like Word2Vec, GloVe, FastText, or BERT.

## Models

- Linear Support Vector Machine using `LinearSVC`
- Decision Tree using `DecisionTreeClassifier`

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
.venv/bin/python "SVM & DT/scripts/create_svm_dt_notebooks.py"
MPLCONFIGDIR=/tmp .venv/bin/jupyter nbconvert --to notebook --execute --inplace "SVM & DT/notebooks/sms_spam_eda_svm_dt.ipynb"
MPLCONFIGDIR=/tmp .venv/bin/jupyter nbconvert --to notebook --execute --inplace "SVM & DT/notebooks/phishing_emails_eda_svm_dt.ipynb"
```

The notebooks load the datasets from the original paths provided for this project.
