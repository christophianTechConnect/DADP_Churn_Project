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
`data/new_customers.csv`, `models/churn_model.joblib` and `models/model_metadata.json`
are included for batch prediction. The model metadata records scikit-learn 1.9.1;
use the version pinned in `requirements.txt` when loading this model.

## Current status

- [x] **Sidebar navigation**: the ☰ button (top left) opens a pop-up sidebar for switching between the
  four pages. Clicking a menu item, ✕, or anywhere outside the sidebar closes it.
- [x] Data Overview page content
- [x] EDA page content
- [x] Single-observation prediction page content (needs the model files to run a prediction; without
  them the page shows a "file model belum tersedia" message)
- [x] Multi-observation prediction page content (CSV upload, validation, model selection, KPIs,
  charts, results table and CSV download)

### Using Prediksi Multi Observasi

1. Open the sidebar and select **Prediksi Multi Observasi**.
2. Choose **Gradient Boosting v1.0.0** and upload a comma-separated UTF-8 CSV.
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
To add another model, add its label, model path and metadata path to `MODEL_REGISTRY`
in `utils/multi_predictions.py`; the dropdown layout needs no changes.

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

There is currently **only one model** (Gradient Boosting v1.0.0: `models/churn_model.joblib` +
`models/model_metadata.json`). The model dropdown on both prediction pages must still be built so that
more models (e.g. Logistic Regression, Random Forest) can be added later without changing the layout.

Models are listed in `MODEL_REGISTRY` in `utils/multi_predictions.py`, shared by both prediction pages.
To add a model, add one entry (label + `.joblib` path + metadata `.json` path); both dropdowns pick it up
automatically. A model whose files are missing still appears in the dropdown, but greyed out and not
selectable.

## Where to add page content

Everything is in [`sesi8/app_final_dashboard.py`](sesi8/app_final_dashboard.py):

| Page | Layout function | Container id |
|---|---|---|
| Data Overview | `page_overview()` | `page-overview` |
| EDA | `page_eda()` | `page-eda` |
| Prediksi Single Observasi | `page_single()` | `page-single` |
| Prediksi Multi Observasi | `page_multi()` | `page-multi` |

1. Replace the `placeholder(...)` return value in your page's function with that page's layout.
2. Add your page's callbacks in section 6 (`CALLBACKS: ISI HALAMAN`).
3. Don't rename the `page-<key>` / `nav-<key>` ids; the sidebar navigation depends on them.
4. All pages are rendered once and only hidden or shown, so component ids must be unique across
   **all** pages (e.g. prefix them with your page key: `eda-...`, `single-...`, `multi-...`).
