# sesi8/app_final_dashboard.py — FINAL DASHBOARD PROJECT (Dash + Plotly + callback + ML inference)
#
# STATUS: Data Overview dan Prediksi Multi Observasi sudah tersedia.
# EDA dan Prediksi Single Observasi masih placeholder; lihat README.md.
#
# Navigasi: tombol ☰ di kiri atas membuka sidebar pop-up berisi 4 halaman:
#   1. Data Overview
#   2. EDA
#   3. Prediksi Single Observasi
#   4. Prediksi Multi Observasi
#
# Cara menambah isi halaman: ganti fungsi page_*() di bagian 3 dengan layout halaman tersebut,
# lalu tambahkan callback-nya di bagian 6. Jangan ubah id "page-<key>" karena dipakai navigasi.
#
# Jalankan dari root project:  python sesi8/app_final_dashboard.py  →  http://127.0.0.1:8053
from pathlib import Path
import sys
import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table, Input, Output, State, callback, ctx, no_update
from dash.exceptions import PreventUpdate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.multi_predictions import MODEL_REGISTRY, predict_upload

# ============ 0. DATA HISTORIS (dimuat sekali saat aplikasi start) ============
RAW = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])  # apa adanya
NUM_COLS = ["tenure_months", "monthly_charges", "total_charges", "num_complaints", "support_calls"]
CAT_COLS = ["region", "segment", "contract_type", "payment_method", "internet_service"]
LABELS = {"customer_id": "ID pelanggan", "signup_date": "Tanggal daftar", "region": "Region",
          "segment": "Segmen", "contract_type": "Jenis kontrak", "payment_method": "Metode pembayaran",
          "internet_service": "Layanan internet", "tenure_months": "Tenure (bulan)",
          "monthly_charges": "Tagihan bulanan", "total_charges": "Total tagihan",
          "num_complaints": "Jumlah keluhan", "support_calls": "Panggilan CS", "churn": "Churn"}
DESKRIPSI = {"customer_id": "ID unik pelanggan", "signup_date": "Tanggal pelanggan mendaftar",
             "region": "Wilayah layanan", "segment": "Segmen bisnis pelanggan",
             "contract_type": "Durasi kontrak", "payment_method": "Cara pembayaran tagihan",
             "internet_service": "Jenis layanan internet", "tenure_months": "Lama berlangganan (bulan)",
             "monthly_charges": "Tagihan per bulan (Rp)", "total_charges": "Akumulasi tagihan (Rp)",
             "num_complaints": "Jumlah keluhan tercatat", "support_calls": "Jumlah panggilan ke CS",
             "churn": "TARGET — 1 = berhenti berlangganan"}

# ============ 1. STYLE ============
C = {"bg": "#F5F7FB", "card": "white", "ink": "#1F2A44", "muted": "#6B7280", "accent": "#2F5DA8"}
CARD = {"background": C["card"], "borderRadius": "12px", "padding": "16px",
        "boxShadow": "0 1px 3px rgba(0,0,0,.08)"}
BTN = {"background": C["accent"], "color": "white", "border": 0, "padding": "6px 14px",
       "borderRadius": "8px", "cursor": "pointer", "fontSize": "20px"}
GRID2 = {"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(420px, 1fr))", "gap": "12px"}
ROW = {"display": "flex", "gap": "12px", "marginBottom": "12px", "flexWrap": "wrap"}
STATUS_COLOR = {"Bertahan": "#4C72B0", "Churn": "#C44E52"}
TABLE_STYLE = dict(style_cell={"padding": "6px", "fontSize": "13px", "textAlign": "left",
                               "fontFamily": "inherit"},
                   style_header={"fontWeight": "bold", "background": "#E9EDF5"},
                   style_table={"overflowX": "auto"})


def kpi(judul, nilai, sub=""):
    return html.Div(style={**CARD, "flex": "1 1 160px"}, children=[
        html.Div(judul, style={"color": C["muted"], "fontSize": "13px"}),
        html.Div(nilai, style={"fontSize": "26px", "fontWeight": 700, "color": C["ink"]}),
        html.Div(sub, style={"color": C["muted"], "fontSize": "12px"})])


def card(*children, **style):
    return html.Div(style={**CARD, **style}, children=list(children))


def empty_fig(msg="Tidak ada data"):
    f = go.Figure()
    f.add_annotation(text=msg, showarrow=False, font={"size": 14, "color": C["muted"]})
    f.update_layout(template="plotly_white", xaxis_visible=False, yaxis_visible=False, height=320)
    return f

# ============ 2. DAFTAR HALAMAN ============
# (key, label menu). key dipakai untuk id: "page-<key>" (kontainer) dan "nav-<key>" (tombol menu).
PAGES = [("overview", "📊  Data Overview"),
         ("eda", "🔍  EDA"),
         ("single", "👤  Prediksi Single Observasi"),
         ("multi", "📁  Prediksi Multi Observasi")]
PAGE_TITLE = {k: t.split("  ", 1)[1] for k, t in PAGES}


# ============ 3. ISI HALAMAN ============
def placeholder(key, deskripsi):
    return html.Div([
        html.H3(PAGE_TITLE[key], style={"margin": "4px 0 12px"}),
        html.Div(style={**CARD, "border": "2px dashed #CBD5E1", "boxShadow": "none",
                        "textAlign": "center", "padding": "48px 16px", "color": C["muted"]},
                 children=[html.Div("🚧 Halaman belum dibuat", style={"fontSize": "18px",
                                                                      "fontWeight": 600}),
                           html.Div(deskripsi, style={"marginTop": "6px"})])])


def page_overview():
    """Profil dataset historis data/customer_churn.csv (statis — tidak butuh callback)."""
    n_rows, n_cols = RAW.shape
    missing = RAW.isna().sum()
    kpis = html.Div(style=ROW, children=[
        kpi("Jumlah baris", f"{n_rows:,}", "pelanggan historis"),
        kpi("Jumlah kolom", f"{n_cols}", f"{len(NUM_COLS)} numerik · {len(CAT_COLS)} kategorikal"),
        kpi("Periode daftar", f"{RAW.signup_date.min():%b %Y} – {RAW.signup_date.max():%b %Y}"),
        kpi("Missing value", f"{int(missing.sum()):,}", f"{missing.sum() / RAW.size:.2%} dari seluruh sel"),
        kpi("Duplikat ID", f"{RAW.customer_id.duplicated().sum():,}"),
        kpi("Churn rate", f"{RAW.churn.mean():.1%}", f"{int(RAW.churn.sum()):,} pelanggan churn")])

    # kamus data: arti, tipe, kelengkapan, dan contoh nilai tiap kolom
    kamus = pd.DataFrame({
        "kolom": RAW.columns,
        "tipe": [("kategorikal" if c in CAT_COLS else "numerik" if c in NUM_COLS
                  else "target" if c == "churn" else "tanggal" if c == "signup_date" else "ID")
                 for c in RAW.columns],
        "deskripsi": [DESKRIPSI.get(c, "") for c in RAW.columns],
        "missing": missing.values,
        "unik": RAW.nunique().values,
        "contoh": [str(RAW[c].dropna().iloc[0]) for c in RAW.columns]})

    stats = RAW[NUM_COLS].describe().T.reset_index().rename(columns={"index": "fitur"})
    stats["fitur"] = stats["fitur"].map(LABELS)
    stats = stats.round(1)

    miss = missing[missing > 0].sort_values()
    if len(miss):
        g_miss = px.bar(x=miss.values, y=miss.index, orientation="h", text_auto=True,
                        template="plotly_white", title="Missing value per kolom",
                        labels={"x": "Jumlah baris kosong", "y": ""},
                        color_discrete_sequence=[C["accent"]])
    else:
        g_miss = empty_fig("Tidak ada missing value")

    tgt = RAW.churn.map({0: "Bertahan", 1: "Churn"}).value_counts().reset_index()
    g_tgt = px.bar(tgt, x="churn", y="count", color="churn", text_auto=True,
                   color_discrete_map=STATUS_COLOR, template="plotly_white",
                   title="Distribusi target (churn)", labels={"churn": "", "count": "Pelanggan"})
    g_tgt.update_layout(showlegend=False)

    per_cat = pd.concat([RAW[c].value_counts().rename_axis("nilai").reset_index().assign(fitur=LABELS[c])
                         for c in CAT_COLS])
    g_cat = px.bar(per_cat, x="count", y="nilai", facet_col="fitur", facet_col_wrap=5,
                   orientation="h", template="plotly_white", height=360,
                   title="Komposisi fitur kategorikal", color_discrete_sequence=[C["accent"]],
                   labels={"count": "", "nilai": ""})
    g_cat.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    g_cat.update_yaxes(matches=None, showticklabels=True)

    preview = RAW.assign(signup_date=RAW.signup_date.dt.strftime("%Y-%m-%d"))
    return html.Div([
        html.H3(PAGE_TITLE["overview"], style={"margin": "4px 0 0"}),
        html.Div("Profil dataset historis data/customer_churn.csv yang dipakai untuk melatih model",
                 style={"color": C["muted"], "marginBottom": "12px"}),
        kpis,
        card(html.H4("Kamus data", style={"marginTop": 0}),
             dash_table.DataTable(data=kamus.to_dict("records"),
                                  columns=[{"name": c.capitalize(), "id": c} for c in kamus.columns],
                                  **TABLE_STYLE), marginBottom="12px"),
        html.Div(style=GRID2, children=[card(dcc.Graph(figure=g_tgt)), card(dcc.Graph(figure=g_miss))]),
        card(dcc.Graph(figure=g_cat), marginTop="12px"),
        card(html.H4("Statistik deskriptif fitur numerik", style={"marginTop": 0}),
             dash_table.DataTable(data=stats.to_dict("records"),
                                  columns=[{"name": c, "id": c} for c in stats.columns], **TABLE_STYLE),
             marginTop="12px"),
        card(html.H4("Pratinjau data mentah", style={"marginTop": 0}),
             dash_table.DataTable(id="overview-raw", data=preview.to_dict("records"),
                                  columns=[{"name": LABELS.get(c, c), "id": c} for c in preview.columns],
                                  page_size=10, sort_action="native", filter_action="native",
                                  **TABLE_STYLE),
             marginTop="12px"),
    ])


def page_eda():
    # TODO: eksplorasi pola churn dengan filter — lihat README.
    return placeholder("eda", "Eksplorasi pola churn pada data historis, lengkap dengan filter.")


def page_single():
    # TODO: form input 1 pelanggan → probabilitas churn + kategori risiko — lihat README.
    # CATATAN MODEL: halaman ini harus punya dropdown "Pilih model". Saat ini baru ada SATU model
    # (Gradient Boosting v1.0.0 → models/churn_model.joblib + models/model_metadata.json), tapi
    # dropdown tetap dibuat agar model lain (mis. Logistic Regression / Random Forest) bisa
    # ditambahkan nanti tanpa mengubah layout.
    return placeholder("single", "Prediksi churn untuk satu pelanggan dari form input. "
                                 "Bisa memilih model (saat ini baru 1 model tersedia).")


def page_multi():
    return html.Div([
        html.H3(PAGE_TITLE["multi"]),
        html.P("Upload CSV pelanggan. Seluruh ringkasan dan hasil pada halaman ini hanya berasal dari file Anda."),
        card(
            html.Label("Pilih model", htmlFor="multi-model"),
            dcc.Dropdown(id="multi-model", clearable=False,
                         options=[{"label": v["label"], "value": k} for k, v in MODEL_REGISTRY.items()],
                         value=next(iter(MODEL_REGISTRY))),
            html.P("CSV UTF-8, pemisah koma. Gunakan kolom fitur seperti pada data/new_customers.csv."),
            dcc.Upload(id="multi-upload", multiple=False, accept=".csv",
                       children=html.Div("Tarik file CSV ke sini atau klik untuk memilih file"),
                       style={"border": "2px dashed #CBD5E1", "borderRadius": "8px",
                              "padding": "28px", "textAlign": "center", "cursor": "pointer"}),
        ),
        dcc.Store(id="multi-results", storage_type="memory"),
        dcc.Loading(html.Div([
            html.Div(id="multi-message", role="status", style={"margin": "12px 0"}),
            html.Div(id="multi-output"),
        ])),
        html.Button("Unduh hasil CSV", id="multi-download-button", n_clicks=0,
                    disabled=True, style={**BTN, "fontSize": "14px", "marginTop": "12px"}),
        dcc.Download(id="multi-download"),
    ])


PAGE_BUILDERS = {"overview": page_overview, "eda": page_eda, "single": page_single, "multi": page_multi}

# ============ 4. LAYOUT: HEADER + SIDEBAR POP-UP ============
DRAWER = {"position": "fixed", "top": 0, "left": 0, "height": "100vh", "width": "280px",
          "background": "white", "boxShadow": "2px 0 12px rgba(0,0,0,.15)", "zIndex": 1001,
          "padding": "20px 16px", "boxSizing": "border-box", "transition": "transform .25s ease"}
DRAWER_OPEN = {**DRAWER, "transform": "translateX(0)"}
DRAWER_CLOSED = {**DRAWER, "transform": "translateX(-100%)"}
BACKDROP = {"position": "fixed", "inset": 0, "background": "rgba(15,23,42,.35)", "zIndex": 1000}
NAV_BTN = {"display": "block", "width": "100%", "textAlign": "left", "padding": "12px 14px",
           "marginBottom": "6px", "border": 0, "borderRadius": "8px", "cursor": "pointer",
           "fontSize": "14px", "background": "transparent", "color": C["ink"]}
NAV_ACTIVE = {**NAV_BTN, "background": "#E8EEF9", "color": C["accent"], "fontWeight": 700}

drawer = html.Div(id="drawer", style=DRAWER_CLOSED, children=[
    html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                    "marginBottom": "16px"}, children=[
        html.H3("Menu", style={"margin": 0}),
        html.Button("✕", id="btn-close", n_clicks=0, title="Tutup menu",
                    style={"border": 0, "background": "none", "fontSize": "20px", "cursor": "pointer"})]),
    *[html.Button(t, id=f"nav-{k}", n_clicks=0, style=NAV_BTN) for k, t in PAGES],
])

app = Dash(__name__, title="Final Dashboard — Customer Churn")
server = app.server                                   # untuk deployment (gunicorn)
app.layout = html.Div(style={"background": C["bg"], "minHeight": "100vh", "padding": "16px",
                             "fontFamily": "Inter, Arial, sans-serif", "color": C["ink"]}, children=[
    dcc.Store(id="page", data=PAGES[0][0]),          # halaman aktif
    html.Div(id="backdrop", n_clicks=0, style={**BACKDROP, "display": "none"}),
    drawer,
    html.Div(style={"display": "flex", "alignItems": "center", "gap": "14px", "marginBottom": "16px"},
             children=[
        html.Button("☰", id="btn-menu", n_clicks=0, title="Buka menu", style=BTN),
        html.Div([html.H2("Customer Churn Intelligence Dashboard", style={"margin": 0}),
                  html.Div(id="page-subtitle", style={"color": C["muted"]})])]),
    # Semua halaman dirender sekali dan hanya disembunyikan/ditampilkan, sehingga isian form,
    # filter, dan file upload tidak hilang saat berpindah halaman.
    *[html.Div(id=f"page-{k}", children=PAGE_BUILDERS[k](), style={"display": "none"})
      for k, _ in PAGES],
])


# ============ 5. CALLBACKS: NAVIGASI ============
@callback(Output("drawer", "style"), Output("backdrop", "style"), Output("page", "data"),
          Input("btn-menu", "n_clicks"), Input("btn-close", "n_clicks"), Input("backdrop", "n_clicks"),
          *[Input(f"nav-{k}", "n_clicks") for k, _ in PAGES], prevent_initial_call=True)
def toggle_drawer(*_):
    """☰ membuka sidebar; ✕, klik di luar sidebar, atau memilih menu akan menutupnya."""
    trig = ctx.triggered_id
    is_open = trig == "btn-menu"
    page = trig.removeprefix("nav-") if trig.startswith("nav-") else no_update
    return (DRAWER_OPEN if is_open else DRAWER_CLOSED,
            {**BACKDROP, "display": "block" if is_open else "none"}, page)


@callback(*[Output(f"page-{k}", "style") for k, _ in PAGES],
          *[Output(f"nav-{k}", "style") for k, _ in PAGES], Output("page-subtitle", "children"),
          Input("page", "data"))
def show_page(page):
    """Menampilkan halaman aktif dan menandai menu yang sedang dipilih."""
    shows = [{"display": "block" if k == page else "none"} for k, _ in PAGES]
    navs = [NAV_ACTIVE if k == page else NAV_BTN for k, _ in PAGES]
    return *shows, *navs, PAGE_TITLE[page]


# ============ 6. CALLBACKS: ISI HALAMAN ============
# TODO: callback tiap halaman ditambahkan di sini oleh anggota tim yang mengerjakannya.


@callback(Output("multi-message", "children"), Output("multi-output", "children"),
          Output("multi-results", "data"), Output("multi-download-button", "disabled"),
          Input("multi-upload", "contents"), Input("multi-model", "value"),
          State("multi-upload", "filename"))
def update_multi(contents, model_id, filename):
    if not contents:
        return "Silakan upload CSV untuk memulai prediksi.", [], None, True
    try:
        result, errors = predict_upload(contents, filename, model_id)
    except Exception:
        logging.exception("Prediksi multi observasi gagal")
        return "Prediksi gagal. Periksa ketersediaan model dan kecocokan versi dependensi di server.", [], None, True
    if errors:
        message = html.Div([html.Strong("File belum valid:"),
                            html.Ul([html.Li(e) for e in errors])], style={"color": "#B91C1C"})
        return message, [], None, True
    risk_colors = {"Rendah": "#1baf7a", "Sedang": "#eda100", "Tinggi": "#e34948"}
    counts = result["risk_category"].value_counts().rename_axis("risk_category").reset_index(name="pelanggan")
    donut = px.pie(counts, names="risk_category", values="pelanggan", color="risk_category",
                   hole=0.5, color_discrete_map=risk_colors, title="Komposisi risiko pelanggan upload")
    donut.update_traces(textinfo="percent+label")
    region_risk = pd.crosstab(result["region"], result["risk_category"]).reindex(
        columns=["Rendah", "Sedang", "Tinggi"], fill_value=0).reset_index()
    bar = px.bar(region_risk.melt(id_vars="region", var_name="risk_category", value_name="pelanggan"),
                 x="region", y="pelanggan", color="risk_category", color_discrete_map=risk_colors,
                 title="Jumlah pelanggan upload per region dan risiko", template="plotly_white")
    # JSON round-trip mengubah NaN pada kolom tambahan menjadi null yang valid untuk Dash.
    import json
    records = json.loads(result.to_json(orient="records", date_format="iso"))
    columns = [{"name": c, "id": c} for c in result.columns]
    output = [
        html.Div(style=ROW, children=[
            kpi("Pelanggan diproses", f"{len(result):,}"),
            kpi("Prediksi churn", f"{int(result.predicted_churn.sum()):,}"),
            kpi("Rata-rata probabilitas", f"{result.churn_probability.mean():.1%}"),
            kpi("Risiko tinggi", f"{int((result.risk_category == 'Tinggi').sum()):,}"),
        ]),
        html.Div(style=GRID2, children=[card(dcc.Graph(figure=donut)), card(dcc.Graph(figure=bar))]),
        card(html.H4("Hasil prediksi seluruh pelanggan upload"),
             dash_table.DataTable(id="multi-table", data=records, columns=columns,
                                  page_size=10, sort_action="native", filter_action="native",
                                  **TABLE_STYLE), marginTop="12px"),
    ]
    return f"{filename}: {len(result):,} pelanggan berhasil diprediksi dengan {MODEL_REGISTRY[model_id]['label']}.", output, records, False


@callback(Output("multi-download", "data"), Input("multi-download-button", "n_clicks"),
          State("multi-results", "data"), prevent_initial_call=True)
def download_multi(n_clicks, records):
    if not n_clicks or not records:
        raise PreventUpdate
    return dcc.send_data_frame(pd.DataFrame(records).to_csv, "hasil_prediksi_multi.csv", index=False)


if __name__ == "__main__":
    app.run(debug=True, port=8053)
