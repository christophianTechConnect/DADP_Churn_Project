# DADP Churn Project — Customer Churn Intelligence Dashboard

Final project dashboard (Dash + Plotly + ML inference) for predicting which customers are likely to churn.

## Running the app

From the project root:

```bash
python -m pip install -r requirements.txt
python sesi8/app_final_dashboard.py
```

Open http://127.0.0.1:8053.

`data/customer_churn.csv` (historical dataset) is in the repo and is loaded when the app starts.
`data/new_customers.csv` (example upload for the Multi page), the three model files in `models/`
and `models/model_metadata.json` are included. The models were saved with scikit-learn 1.9.1;
use the versions pinned in `requirements.txt` when loading them.

> **Pulled the top-3 models update?** Run `python -m pip install -r requirements.txt` again.
> It adds `imbalanced-learn`, which the *Logistic Regression + Oversampling* model needs; without
> it, selecting that model shows an error.

## Current status

- [x] **Sidebar navigation**: the ☰ button (top left) opens a pop-up sidebar for switching between the
  five pages (including Team). Clicking a menu item, ✕, or anywhere outside the sidebar closes it.
- [x] Data Overview page content
- [x] EDA page content
- [x] Single-observation prediction page content (needs the model files to run a prediction; without
  them the page shows a "file model belum tersedia" message)
- [x] Multi-observation prediction page content (CSV upload, validation, model selection, KPIs,
  charts, results table and CSV download)
- [x] Team page: lists everyone who has committed to the repo

### Team page

The **Team** page lists the repo's contributors from the git history (`git log`, merge commits
excluded), read once when the app starts. It uses git history rather than GitHub's contributors API
because GitHub only counts commits whose email is linked to a GitHub account, so it misses some team
members. If one person appears twice (different name or email on different machines), add a line to
[`.mailmap`](.mailmap) to merge them. The page needs the `.git` folder; without it, it shows a
"Riwayat git tidak tersedia" message instead.

### Using Prediksi Multi Observasi

1. Open the sidebar and select **Prediksi Multi Observasi**.
2. Choose a model (see [Models](#note-on-model-selection)) and upload a comma-separated UTF-8 CSV.
   Use `data/new_customers.csv` as an example; it is not loaded automatically.
3. Required features: `tenure_months`, `monthly_charges`, `total_charges`,
   `num_complaints`, `support_calls`, `region`, `segment`, `contract_type`,
   `payment_method`, `internet_service`. `customer_id` and other extra columns are optional
   and preserved in the results.
4. Fix any missing columns, unknown categories, empty feature values, or invalid numbers
   reported by validation, then upload again. An invalid upload clears the previous results.
5. Review the uploaded customers' KPIs, risk composition, region chart and results table.
   **Unduh hasil CSV** exports all predicted rows, including probability, class and risk category.

Only CSV is supported in this implementation; Excel is optional in the assignment.
All page results remain empty until upload. Changing models reruns prediction on the same file.
To add another model, see [Note on model selection](#note-on-model-selection); the dropdown layout
needs no changes.

Run validation and callback tests from the project root:

```bash
python -m unittest discover -s tests -v
```

## Requirements

### General

- The left sidebar opens as a pop-up and lets users navigate between **Data Overview**, **EDA**,
  **Prediksi Single Observasi** and **Prediksi Multi Observasi**. *(done)*
- Each page must have complete content. The items listed below are the minimum for each page.

### 1. Data Overview *(done)*

Profile of the historical dataset (`data/customer_churn.csv`):

- Summary KPIs: number of rows and columns, signup period, missing values, churn rate
- Data dictionary: column name, type, description, missing count, unique values
- Target (churn) distribution and missing values per column
- Descriptive statistics for numeric features
- Raw data preview table

### 2. EDA *(done)*

Exploration of churn patterns in the historical data. Every KPI, chart and insight on the page
follows the filters.

- Filters in one row: region, segment, contract type (empty = all), tenure range, and a
  **Reset filter** button
- KPIs: customers, churn rate (compared with the full dataset), churned customers, median tenure and
  average complaints for churned vs. retained customers
- Churn rate by categorical feature (a dropdown picks the feature), with the average as a reference
  line, plus a churn-rate heatmap of segment × contract type
- Numeric features (a dropdown picks the feature): distribution by churn status (% within each
  status, with box plots), and churn rate per value band. Band edges are fixed from the full
  dataset (tenure 1–6/7–12/13–24/25–48/49–72 months, charge quintiles, complaint and call counts),
  so the bands stay the same when filters change.
- Churn rate trend by signup quarter
- Correlation of each numeric feature with churn, and a full correlation matrix
- **Insight utama**: written insights recalculated from the filtered data (contract type, tenure,
  complaints, highest-churn group, strongest correlation). Groups with fewer than 30 customers are
  skipped.

Callbacks: `update_eda_overview`, `update_eda_categorical`, `update_eda_numeric` and
`reset_eda_filters` (section 6b). Component ids use the `eda-` prefix.

### 3. Prediksi Single Observasi *(done)*

- A form for entering one customer's details → churn probability, predicted class, risk category
  and a recommended action
- **Model selection**: a dropdown to choose which model makes the prediction

### 4. Prediksi Multi Observasi *(done)*

- **File upload** (CSV, optionally Excel) of multiple customers
- All charts, KPIs and tables on this page show **only the data from the uploaded file**, not the
  historical dataset. Nothing is shown before a file is uploaded.
- Validate the file (required columns, unknown category values) and show clear error messages
- Results table with churn probability and risk category, plus a way to download the results
- **Model selection**: a dropdown to choose which model makes the predictions

### Note on model selection

Both prediction pages offer the **top 3 models by cross-validated ROC-AUC** (5-fold stratified), listed
in [`models/top_models_registry.json`](models/top_models_registry.json). The dropdown is ordered by rank
and the #1 model is selected by default.

| Rank | Model | File | CV ROC-AUC | Holdout ROC-AUC | Holdout recall |
|---|---|---|---|---|---|
| 1 | Logistic Regression | `churn_model_logreg.joblib` | 0.800 ± 0.013 | 0.796 | 0.524 |
| 2 | Logistic Regression + Oversampling | `churn_model_logreg_ros.joblib` | 0.799 ± 0.013 | 0.794 | 0.719 |
| 3 | Gradient Boosting (original final model) | `churn_model.joblib` | 0.792 ± 0.012 | 0.789 | 0.552 |

The three are very close on ROC-AUC. The oversampling model catches the most churners (highest recall)
at the cost of more false alarms, so it predicts higher churn probabilities overall.

All three use the same input contract, `models/model_metadata.json` (features, categories, threshold
0.5, risk bins). The oversampling model needs `imbalanced-learn` (in `requirements.txt`).

To add a model: put its `.joblib` file in `models/` and add one entry to `top_models_registry.json`
(`rank`, `key`, `name`, `file`, metrics). Both dropdowns pick it up automatically
(`MODEL_REGISTRY` in `utils/multi_predictions.py` is built from that file). A model whose file is
missing still appears in the dropdown, but greyed out and not selectable.

Each model's holdout ROC-AUC shown on the Single page comes from its own entry in
`top_models_registry.json`, not from the shared `model_metadata.json` (which holds Gradient Boosting's
numbers).

**Default model.** The default is #1 Logistic Regression. The business insights in
[`sesi8/LAPORAN_FINAL_PROJECT.md`](sesi8/LAPORAN_FINAL_PROJECT.md) section 11 (and
`sesi8/insight_bisnis.py`) are computed with **#3 Gradient Boosting**, the original final model. The
three models give different probabilities for the same customer: with the Single page's default form
values they give 89.3% (#1), 93.5% (#2) and 86.0% (#3), all "Tinggi". The oversampling model tends to
score customers higher.

`.joblib` files are pickles: loading one can run code, so only add model files from a trusted source.
The two Logistic Regression files were checked before being added: they only reference scikit-learn,
numpy, imbalanced-learn and joblib. The Gradient Boosting file from the same download produced
identical predictions to the one already in the repo, so the repo's copy was kept.

## Code layout

The app is in [`sesi8/app_final_dashboard.py`](sesi8/app_final_dashboard.py):

| Page | Layout function | Container id | Callbacks (section 6) |
|---|---|---|---|
| Data Overview | `page_overview()` | `page-overview` | none (static) |
| EDA | `page_eda()` | `page-eda` | `update_eda_overview`, `update_eda_categorical`, `update_eda_numeric`, `reset_eda_filters` |
| Prediksi Single Observasi | `page_single()` | `page-single` | `run_single_prediction` |
| Prediksi Multi Observasi | `page_multi()` | `page-multi` | `update_multi`, `download_multi` |
| Team | `page_team()` | `page-team` | none (static) |

Helpers: [`utils/model_utils.py`](utils/model_utils.py) (load model, validate, predict),
[`utils/multi_predictions.py`](utils/multi_predictions.py) (model registry, CSV upload),
[`utils/contributors.py`](utils/contributors.py) (Team page). The written project report is
[`sesi8/LAPORAN_FINAL_PROJECT.md`](sesi8/LAPORAN_FINAL_PROJECT.md).

To add a page:

1. Add `(key, label)` to `PAGES` (section 2), write `page_<key>()` (section 3) and register it in
   `PAGE_BUILDERS`.
2. Add its callbacks in section 6 (`CALLBACKS: ISI HALAMAN`).
3. Don't rename the `page-<key>` / `nav-<key>` ids; the sidebar navigation depends on them.
4. All pages are rendered once and only hidden or shown, so component ids must be unique across
   **all** pages (prefix them with the page key: `eda-...`, `single-...`, `multi-...`, `team-...`).
