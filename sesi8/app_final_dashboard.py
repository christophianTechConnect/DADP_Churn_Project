# sesi8/app_final_dashboard.py — FINAL DASHBOARD PROJECT (Dash + Plotly + callback + ML inference)
#
# STATUS: navigasi dan keempat halaman (Data Overview, EDA, Prediksi Single, Prediksi Multi)
# sudah dibuat. Lihat README.md bagian "Requirements" untuk requirement tiap halaman.
#
# Navigasi: tombol ☰ di kiri atas membuka sidebar pop-up berisi 5 halaman:
#   1. Data Overview
#   2. EDA
#   3. Prediksi Single Observasi
#   4. Prediksi Multi Observasi
#   5. Team (kontributor repo)
#
# Cara menambah isi halaman: ganti fungsi page_*() di bagian 3 dengan layout halaman tersebut,
# lalu tambahkan callback-nya di bagian 6. Jangan ubah id "page-<key>" karena dipakai navigasi.
#
# Jalankan dari root project:  python sesi8/app_final_dashboard.py  →  http://127.0.0.1:8053
import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table, Input, Output, State, callback, ctx, no_update
from dash.exceptions import PreventUpdate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.model_utils import validate_input, predict  # noqa: E402
# MODEL_REGISTRY ada di utils/multi_predictions.py dan dipakai KEDUA halaman prediksi.
from utils.multi_predictions import MODEL_REGISTRY, get_model_bundle, predict_upload  # noqa: E402
from utils.contributors import load_contributors  # noqa: E402


# ============ 0a. DAFTAR MODEL (dipakai dropdown "Pilih model" di halaman prediksi) ============
# Menambah model baru cukup dengan menambah satu entri di MODEL_REGISTRY
# (utils/multi_predictions.py); layout halaman tidak perlu diubah.
def model_available(key):
    cfg = MODEL_REGISTRY[key]
    return cfg["model"].is_file() and cfg["metadata"].is_file()


def model_options():
    """Opsi dropdown model; model yang file-nya belum ada tetap tampil tapi tidak bisa dipilih."""
    return [{"label": cfg["label"] + ("" if model_available(k) else "  (file model belum tersedia)"),
             "value": k, "disabled": not model_available(k)} for k, cfg in MODEL_REGISTRY.items()]


def default_model():
    return next((k for k in MODEL_REGISTRY if model_available(k)), next(iter(MODEL_REGISTRY)))


def get_model(key):
    """Memuat model + metadata sekali (cache lru di get_model_bundle), dipakai halaman Single & Multi."""
    return get_model_bundle(key)

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
RISK_COLOR = {"Rendah": "#55A868", "Sedang": "#DD8452", "Tinggi": "#C44E52"}
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
         ("multi", "📁  Prediksi Multi Observasi"),
         ("team", "👥  Team")]
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


# ---------- 3b. EDA ----------
MIN_N = 30                                             # kelompok < 30 pelanggan terlalu kecil untuk insight
CONTRACT_ORDER = ["Bulanan", "Tahunan", "Dua Tahun"]
STATUS_ORDER = ["Bertahan", "Churn"]
CAPTION = {"color": C["muted"], "fontSize": "13px", "fontWeight": 600, "marginTop": "8px"}


def _short_rp(x):
    return f"{x / 1e6:.1f}jt".replace(".", ",") if x >= 1e6 else f"{x / 1e3:.0f}rb"


def _quintile_bins(col):
    q = RAW[col].quantile([0, .2, .4, .6, .8, 1]).round(-3).tolist()
    return ([-float("inf"), *q[1:-1], float("inf")],
            [f"{_short_rp(a)}–{_short_rp(b)}" for a, b in zip(q[:-1], q[1:])])


# Kelompok fitur numerik untuk grafik "churn rate per kelompok". Batasnya dihitung sekali dari data
# penuh, sehingga kelompoknya tetap sama walaupun filter berubah.
NUM_BANDS = {
    "tenure_months": ([0, 6, 12, 24, 48, 72], ["1–6", "7–12", "13–24", "25–48", "49–72"]),
    "monthly_charges": _quintile_bins("monthly_charges"),
    "total_charges": _quintile_bins("total_charges"),
    "num_complaints": ([-1, 0, 1, 2, 3, float("inf")], ["0", "1", "2", "3", "4+"]),
    "support_calls": ([-1, 0, 1, 2, 3, 4, 5, float("inf")], ["0", "1", "2", "3", "4", "5", "6+"]),
}


def _filter_dd(id_, judul, options):
    return html.Div(style={"flex": "1 1 180px"}, children=[
        html.Label(judul, style={"fontWeight": 600, "fontSize": "13px"}),
        dcc.Dropdown(id=id_, options=options, multi=True, placeholder="Semua")])


def _card_head(judul, control):
    return html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                           "gap": "12px", "flexWrap": "wrap"},
                    children=[html.H4(judul, style={"margin": 0}), control])


def _feature_dd(id_, cols, value):
    return dcc.Dropdown(id=id_, options=[{"label": LABELS[c], "value": c} for c in cols], value=value,
                        clearable=False, style={"width": "210px"})


def page_eda():
    """Eksplorasi pola churn pada data historis. Semua isi diperbarui oleh callback update_eda_*."""
    filters = card(html.Div(
        style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "flex-end"}, children=[
            _filter_dd("eda-region", "Region", sorted(RAW.region.unique())),
            _filter_dd("eda-segment", "Segmen", sorted(RAW.segment.unique())),
            _filter_dd("eda-contract", "Jenis kontrak", CONTRACT_ORDER),
            html.Div(style={"flex": "2 1 280px"}, children=[
                html.Label("Tenure (bulan)", style={"fontWeight": 600, "fontSize": "13px"}),
                dcc.RangeSlider(id="eda-tenure", min=1, max=72, step=1, value=[1, 72],
                                marks={m: str(m) for m in (1, 12, 24, 36, 48, 60, 72)},
                                tooltip={"placement": "bottom"})]),
            html.Button("Reset filter", id="eda-reset", n_clicks=0,
                        style={**BTN, "fontSize": "13px", "padding": "9px 14px"})]),
        marginBottom="12px")

    return html.Div([
        html.H3(PAGE_TITLE["eda"], style={"margin": "4px 0 0"}),
        html.Div("Eksplorasi pola churn pada data historis — KPI, grafik, dan insight mengikuti filter",
                 style={"color": C["muted"], "marginBottom": "12px"}),
        filters,
        html.Div(id="eda-kpis", style=ROW),
        card(html.H4("Insight utama", style={"marginTop": 0}),
             html.Div(id="eda-insights", style={"lineHeight": 1.6}),
             html.Div(f"Dihitung ulang setiap filter berubah. Kelompok dengan kurang dari {MIN_N} pelanggan "
                      "diabaikan. Korelasi menunjukkan keterkaitan, bukan sebab-akibat.",
                      style={"color": C["muted"], "fontSize": "12px"}),
             marginBottom="12px"),
        html.Div(style=GRID2, children=[
            card(_card_head("Churn rate per fitur kategorikal",
                            _feature_dd("eda-cat", CAT_COLS, "contract_type")),
                 dcc.Graph(id="eda-cat-rate")),
            card(html.H4("Churn rate: segmen × jenis kontrak", style={"margin": 0}),
                 dcc.Graph(id="eda-heatmap"))]),
        card(_card_head("Fitur numerik vs churn", _feature_dd("eda-num", NUM_COLS, "tenure_months")),
             html.Div(style=GRID2, children=[
                 html.Div([html.Div("Distribusi per status (% dari tiap status)", style=CAPTION),
                           dcc.Graph(id="eda-num-dist")]),
                 html.Div([html.Div("Churn rate per kelompok nilai", style=CAPTION),
                           dcc.Graph(id="eda-num-rate")])]),
             marginTop="12px"),
        card(html.H4("Tren churn rate per kuartal pendaftaran", style={"margin": 0}),
             dcc.Graph(id="eda-trend"), marginTop="12px"),
        html.Div(style={**GRID2, "marginTop": "12px"}, children=[
            card(html.H4("Korelasi fitur numerik dengan churn", style={"margin": 0}),
                 dcc.Graph(id="eda-corr")),
            card(html.H4("Matriks korelasi", style={"margin": 0}), dcc.Graph(id="eda-corr-matrix"))]),
    ])


# --- Prediksi Single Observasi ---
# Nilai awal form = profil contoh pelanggan baru berkontrak bulanan (agar demo langsung bisa dijalankan).
SINGLE_DEFAULT = {"region": "Sulawesi", "segment": "UMKM", "contract_type": "Bulanan",
                  "payment_method": "Transfer Bank", "internet_service": "Fiber",
                  "tenure_months": 6, "monthly_charges": 450000, "total_charges": None,
                  "num_complaints": 3, "support_calls": 5}
SINGLE_STEP = {"tenure_months": 1, "monthly_charges": 1000, "total_charges": 1000,
               "num_complaints": 1, "support_calls": 1}
SINGLE_FIELDS = CAT_COLS + NUM_COLS
AKSI_RISIKO = {
    "Tinggi": ("Prioritas retensi segera",
               "Hubungi pelanggan dalam 48 jam melalui tim retensi dan tawarkan insentif personal "
               "(diskon, upgrade layanan, atau bonus kuota)."),
    "Sedang": ("Pantau dan lakukan engagement",
               "Masukkan ke kampanye engagement bulan ini (survei kepuasan, promo upgrade) dan pantau "
               "keluhan serta panggilan CS-nya."),
    "Rendah": ("Pertahankan kualitas layanan",
               "Tidak perlu intervensi khusus. Pertahankan kualitas layanan dan tawarkan program "
               "loyalitas atau upsell."),
}
LABEL_STYLE = {"fontSize": "13px", "color": C["muted"], "marginBottom": "4px", "display": "block"}


def rupiah(x):
    return f"Rp{x:,.0f}".replace(",", ".")


def single_field(col):
    """Satu baris form: label + dropdown (kategorikal) atau input angka (numerik)."""
    if col in CAT_COLS:
        ctrl = dcc.Dropdown(id=f"single-{col}", options=sorted(RAW[col].dropna().unique()),
                            value=SINGLE_DEFAULT[col], clearable=False)
    else:
        ctrl = dcc.Input(id=f"single-{col}", type="number", min=0, step=SINGLE_STEP[col],
                         value=SINGLE_DEFAULT[col],
                         placeholder="otomatis" if col == "total_charges" else "",
                         style={"width": "100%", "padding": "7px 8px", "boxSizing": "border-box",
                                "border": "1px solid #ccc", "borderRadius": "4px"})
    label = LABELS[col] + (" (opsional)" if col == "total_charges" else "")
    return html.Div([html.Label(label, htmlFor=f"single-{col}", style=LABEL_STYLE), ctrl],
                    style={"marginBottom": "10px"})


def gauge_fig(p, meta):
    """Gauge probabilitas churn, berwarna sesuai rentang kategori risiko + garis threshold."""
    steps = [{"range": [lo * 100, hi * 100], "color": RISK_COLOR.get(lbl, "#DDD") + "40"}
             for lbl, (lo, hi) in meta["risk_bins"].items()]
    f = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "valueformat": ".1f"},
        gauge={"axis": {"range": [0, 100], "ticksuffix": "%"}, "bar": {"color": C["ink"], "thickness": .3},
               "steps": steps,
               "threshold": {"line": {"color": C["accent"], "width": 3}, "thickness": .9,
                             "value": meta["decision_threshold"] * 100}}))
    f.update_layout(height=260, margin={"t": 30, "b": 10, "l": 30, "r": 30}, template="plotly_white",
                    title={"text": f"Probabilitas churn (garis biru = threshold "
                                   f"{meta['decision_threshold']:.0%})", "font": {"size": 14}})
    return f


def faktor_risiko(row):
    """Saran spesifik berdasarkan karakteristik pelanggan (temuan insight bisnis sesi 8)."""
    saran = []
    if row["contract_type"] == "Bulanan":
        saran.append("Kontrak Bulanan: tawarkan migrasi ke kontrak Tahunan dengan insentif diskon.")
    if row["tenure_months"] < 12:
        saran.append(f"Pelanggan baru ({row['tenure_months']:.0f} bulan): lakukan kontak onboarding proaktif "
                     "selama 12 bulan pertama.")
    if row["num_complaints"] >= 2:
        saran.append(f"{row['num_complaints']:.0f} keluhan tercatat: pastikan diselesaikan maksimal 48 jam (SLA).")
    if row["support_calls"] >= 5:
        saran.append(f"{row['support_calls']:.0f} kali menghubungi CS: periksa kemungkinan masalah teknis berulang.")
    if row["payment_method"] == "Transfer Bank":
        saran.append("Pembayaran via Transfer Bank: dorong autodebit atau e-wallet untuk mengurangi friksi.")
    return saran


def single_placeholder(msg, icon="🧮", color=None):
    return html.Div(style={**CARD, "border": "2px dashed #CBD5E1", "boxShadow": "none",
                           "textAlign": "center", "padding": "56px 16px", "color": color or C["muted"]},
                    children=[html.Div(icon, style={"fontSize": "36px"}), html.Div(msg)])


def page_single():
    """Form input satu pelanggan → probabilitas churn, kelas, kategori risiko, dan rekomendasi."""
    tersedia = [k for k in MODEL_REGISTRY if model_available(k)]
    model_card = card(
        html.Label("Pilih model", htmlFor="single-model", style={**LABEL_STYLE, "fontWeight": 600}),
        dcc.Dropdown(id="single-model", options=model_options(), value=default_model(), clearable=False),
        html.Div(f"{len(tersedia)} dari {len(MODEL_REGISTRY)} model tersedia",
                 style={"fontSize": "12px", "color": C["muted"], "marginTop": "6px"}),
        marginBottom="12px")
    form_card = card(
        html.H4("Data pelanggan", style={"marginTop": 0}),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "columnGap": "12px"},
                 children=[single_field(c) for c in SINGLE_FIELDS]),
        html.Button("Prediksi churn", id="single-btn", n_clicks=0,
                    style={**BTN, "fontSize": "15px", "width": "100%", "padding": "10px"}))
    return html.Div([
        html.H3(PAGE_TITLE["single"], style={"margin": "4px 0 0"}),
        html.Div("Isi data satu pelanggan, pilih model, lalu klik Prediksi churn untuk melihat "
                 "probabilitas, kategori risiko, dan tindakan yang disarankan.",
                 style={"color": C["muted"], "marginBottom": "12px"}),
        html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "flex-start"},
                 children=[
            html.Div([model_card, form_card], style={"flex": "1 1 340px", "maxWidth": "460px"}),
            dcc.Loading(html.Div(id="single-result", children=single_placeholder(
                "Hasil prediksi akan muncul di sini setelah tombol Prediksi churn diklik.")),
                parent_style={"flex": "2 1 480px", "minWidth": 0}),
        ]),
    ])


def page_multi():
    return html.Div([
        html.H3(PAGE_TITLE["multi"]),
        html.P("Upload CSV pelanggan. Seluruh ringkasan dan hasil pada halaman ini hanya berasal dari file Anda."),
        card(
            html.Label("Pilih model", htmlFor="multi-model"),
            dcc.Dropdown(id="multi-model", clearable=False,
                         options=model_options(), value=default_model()),
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


REPO_URL = "https://github.com/christophianTechConnect/DADP_Churn_Project"


def team_card(row):
    """Kartu satu kontributor: inisial, nama, link GitHub, jumlah commit, dan commit terakhir."""
    inisial = "".join(w[0] for w in row["name"].replace("-", " ").split()[:2]).upper()
    nama = (html.A(row["name"], href=f"https://github.com/{row['github']}", target="_blank",
                   style={"color": C["ink"], "textDecoration": "none"})
            if row["github"] else row["name"])
    return html.Div(style={**CARD, "display": "flex", "gap": "14px", "alignItems": "flex-start"}, children=[
        html.Div(inisial, style={"flex": "0 0 48px", "height": "48px", "borderRadius": "50%",
                                 "background": "#E8EEF9", "color": C["accent"], "fontWeight": 700,
                                 "display": "flex", "alignItems": "center", "justifyContent": "center"}),
        html.Div(style={"minWidth": 0}, children=[
            html.Div(nama, style={"fontWeight": 700, "fontSize": "16px"}),
            html.Div(f"@{row['github']}" if row["github"] else "", style={"color": C["muted"],
                                                                         "fontSize": "12px"}),
            html.Div(f"{row['commits']} commit · {row['first_commit']} – {row['last_commit']}",
                     style={"color": C["muted"], "fontSize": "13px", "margin": "4px 0"}),
            html.Ul([html.Li(s) for s in row["recent"]],
                    style={"margin": 0, "paddingLeft": "18px", "fontSize": "13px"})])])


def page_team():
    """Kontributor repo GitHub, dihitung dari riwayat git saat aplikasi start."""
    team = load_contributors(ROOT)
    head = [html.H3(PAGE_TITLE["team"], style={"margin": "4px 0 0"}),
            html.Div(["Kontributor ", html.A("repo GitHub", href=REPO_URL, target="_blank"),
                      " dihitung dari riwayat commit (tanpa commit merge)."],
                     style={"color": C["muted"], "marginBottom": "12px"})]
    if team.empty:
        return html.Div([*head, card("Riwayat git tidak tersedia, jadi daftar kontributor tidak bisa "
                                     "ditampilkan. Jalankan aplikasi dari folder hasil git clone.",
                                     color=C["muted"])])
    return html.Div([
        *head,
        html.Div(style=ROW, children=[
            kpi("Kontributor", f"{len(team)}"),
            kpi("Total commit", f"{int(team.commits.sum())}", "tanpa commit merge"),
            kpi("Periode", f"{team.first_commit.min()} – {team.last_commit.max()}")]),
        html.Div(id="team-list", style=GRID2, children=[team_card(r) for _, r in team.iterrows()]),
    ])


PAGE_BUILDERS = {"overview": page_overview, "eda": page_eda, "single": page_single, "multi": page_multi,
                 "team": page_team}

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
# TODO: callback halaman Prediksi ditambahkan di sini oleh anggota tim yang mengerjakannya.

# ---------- 6b. EDA ----------
EDA_FILTERS = [Input("eda-region", "value"), Input("eda-segment", "value"),
               Input("eda-contract", "value"), Input("eda-tenure", "value")]
NO_DATA = "Tidak ada pelanggan yang cocok — longgarkan filter"
RATE_HOVER = "Churn rate: %{customdata[2]:.1%}<br>%{customdata[1]:,} dari %{customdata[0]:,} pelanggan"


def eda_filter(region, segment, contract, tenure):
    """Filter kosong berarti 'semua'."""
    d = RAW
    if region: d = d[d.region.isin(region)]
    if segment: d = d[d.segment.isin(segment)]
    if contract: d = d[d.contract_type.isin(contract)]
    if tenure: d = d[d.tenure_months.between(*tenure)]
    return d


def churn_rates(d, col, min_n=1):
    g = (d.groupby(col, observed=False).churn.agg(rate="mean", n="size", churned="sum").reset_index())
    return g[g.n >= min_n]


def corr_with_churn(d):
    """Korelasi (Pearson / point-biserial) tiap fitur numerik dengan churn; kosong jika tak terhitung."""
    cols = [c for c in NUM_COLS if d[c].nunique() > 1]
    if d.churn.nunique() < 2 or not cols:
        return pd.Series(dtype=float)
    return d[cols].corrwith(d.churn).dropna()


def fig_style(fig, height=340, **kw):
    fig.update_layout(template="plotly_white", height=height, margin=dict(t=30, b=10, l=10, r=10),
                      font_family="Inter, Arial, sans-serif", **kw)
    return fig


def add_avg_line(fig, rate, horizontal):
    line = dict(line_dash="dash", line_color=C["muted"], line_width=1,
                annotation_text=f"rata-rata {rate:.1%}", annotation_font_color=C["muted"])
    (fig.add_hline(y=rate, annotation_position="top right", **line) if horizontal
     else fig.add_vline(x=rate, annotation_position="top", **line))


def _fmt(x, spec):
    return "–" if pd.isna(x) else format(x, spec)


def eda_kpis(d):
    n, churned = len(d), int(d.churn.sum())
    by = d.groupby("churn")
    ten, comp = by.tenure_months.median(), by.num_complaints.mean()
    return [kpi("Pelanggan", f"{n:,}", f"{n / len(RAW):.0%} dari seluruh data"),
            kpi("Churn rate", f"{d.churn.mean():.1%}", f"keseluruhan data: {RAW.churn.mean():.1%}"),
            kpi("Pelanggan churn", f"{churned:,}", f"{n - churned:,} bertahan"),
            kpi("Median tenure", f"{_fmt(ten.get(1), '.0f')} vs {_fmt(ten.get(0), '.0f')} bln",
                "churn vs bertahan"),
            kpi("Rata-rata keluhan", f"{_fmt(comp.get(1), '.1f')} vs {_fmt(comp.get(0), '.1f')}",
                "churn vs bertahan")]


def eda_insights(d):
    """Insight tertulis yang dihitung dari data terfilter (kelompok < MIN_N dilewati)."""
    out, n, rate, overall = [], len(d), d.churn.mean(), RAW.churn.mean()
    diff = rate - overall
    if n == len(RAW):
        banding = "ini adalah seluruh data historis"
    elif abs(diff) < 0.005:
        banding = f"setara dengan keseluruhan data ({overall:.1%})"
    else:
        banding = f"{abs(diff) * 100:.1f} poin {'di atas' if diff > 0 else 'di bawah'} keseluruhan data ({overall:.1%})"
    out.append(f"Churn rate **{rate:.1%}** ({int(d.churn.sum()):,} dari {n:,} pelanggan) — {banding}.")

    g = churn_rates(d, "contract_type", MIN_N)
    if len(g) >= 2:
        hi, lo = g.loc[g.rate.idxmax()], g.loc[g.rate.idxmin()]
        kali = f", {hi.rate / lo.rate:.1f}× lipat" if lo.rate > 0 else ""
        teks = (f"Kontrak **{hi.contract_type}** paling rentan: churn {hi.rate:.1%} vs {lo.rate:.1%} "
                f"pada kontrak {lo.contract_type}{kali}.")
        if CONTRACT_ORDER.index(lo.contract_type) > CONTRACT_ORDER.index(hi.contract_type):
            teks += " Kontrak yang lebih panjang berkaitan dengan churn yang jauh lebih rendah."
        out.append(teks)

    baru, lama = d[d.tenure_months <= 12], d[d.tenure_months > 24]
    if len(baru) >= MIN_N and len(lama) >= MIN_N:
        rb, rl = baru.churn.mean(), lama.churn.mean()
        out.append(f"Pelanggan baru (tenure ≤ 12 bln) churn **{rb:.1%}**, dibanding {rl:.1%} untuk tenure "
                   f"> 24 bln — " + ("risiko terbesar ada di tahun pertama berlangganan."
                                     if rb > rl + 0.05 else "perbedaannya kecil pada filter ini."))

    nol, banyak = d[d.num_complaints == 0], d[d.num_complaints >= 3]
    if len(nol) >= MIN_N and len(banyak) >= MIN_N:
        r0, r3 = nol.churn.mean(), banyak.churn.mean()
        out.append(f"Pelanggan dengan ≥ 3 keluhan churn **{r3:.1%}**, dibanding {r0:.1%} tanpa keluhan — "
                   + ("keluhan adalah sinyal peringatan dini churn." if r3 > r0 + 0.05
                      else "keluhan tidak banyak membedakan churn pada filter ini."))

    best = None
    for col in ["segment", "region", "payment_method", "internet_service"]:
        g = churn_rates(d, col, MIN_N)
        if len(g) >= 2:
            top = g.loc[g.rate.idxmax()]
            if best is None or top.rate > best[2]:
                best = (col, top[col], top.rate, int(top.n))
    if best:
        out.append(f"Di luar jenis kontrak, churn tertinggi ada pada **{LABELS[best[0]]} {best[1]}** "
                   f"({best[2]:.1%}, n = {best[3]:,}).")

    r = corr_with_churn(d) if n >= MIN_N else pd.Series(dtype=float)
    if len(r):
        top = r.abs().idxmax()
        v = r[top]
        kuat = "kuat" if abs(v) >= 0.5 else "sedang" if abs(v) >= 0.3 else "lemah"
        arah = "makin besar" if v > 0 else "makin kecil"
        out.append(f"Fitur numerik paling terkait dengan churn: **{LABELS[top]}** (r = {v:+.2f}, korelasi "
                   f"{kuat}) — makin tinggi nilainya, {arah} kecenderungan churn.")
        if {"tenure_months", "total_charges"} <= set(r.index):
            rt = d.tenure_months.corr(d.total_charges)
            if rt >= 0.7:
                out.append(f"**Total tagihan** sangat berkorelasi dengan tenure (r = {rt:.2f}) karena merupakan "
                           "akumulasi tagihan bulanan — informasinya banyak tumpang tindih.")
    return dcc.Markdown("\n".join(f"- {s}" for s in out))


def fig_heatmap(d):
    segs = sorted(d.segment.unique())
    cons = [c for c in CONTRACT_ORDER if c in set(d.contract_type)]
    pv = d.groupby(["segment", "contract_type"]).churn.agg(["mean", "size"])
    rate = pv["mean"].unstack().reindex(index=segs, columns=cons)
    n = pv["size"].unstack().reindex(index=segs, columns=cons).fillna(0).astype(int)
    text = [[f"{r:.0%}<br>n={k}" if k else "" for r, k in zip(rr, kk)]
            for rr, kk in zip(rate.values, n.values)]
    fig = go.Figure(go.Heatmap(
        z=rate.values, x=cons, y=segs, text=text, texttemplate="%{text}", customdata=n.values,
        colorscale="Reds", zmin=0, zmax=1, xgap=2, ygap=2,
        colorbar=dict(title="Churn", tickformat=".0%", thickness=12),
        hovertemplate="<b>%{y} · %{x}</b><br>Churn rate: %{z:.1%}<br>n = %{customdata:,}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return fig_style(fig)


def fig_trend(d):
    g = churn_rates(d.assign(kuartal=d.signup_date.dt.to_period("Q").astype(str)), "kuartal")
    fig = px.line(g, x="kuartal", y="rate", markers=True, custom_data=["n", "churned", "rate"],
                  color_discrete_sequence=[C["accent"]],
                  labels={"kuartal": "Kuartal pendaftaran", "rate": "Churn rate"})
    fig.update_traces(line_width=2, marker_size=8, hovertemplate=f"<b>%{{x}}</b><br>{RATE_HOVER}<extra></extra>")
    add_avg_line(fig, d.churn.mean(), horizontal=True)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return fig_style(fig, height=300)


def fig_corr(d):
    r = corr_with_churn(d).sort_values()
    if r.empty:
        return empty_fig("Korelasi tidak bisa dihitung untuk filter ini")
    pos, neg = "Searah churn (+)", "Berlawanan churn (−)"
    cr = pd.DataFrame({"fitur": r.index.map(LABELS), "r": r.values,
                       "arah": [pos if v >= 0 else neg for v in r.values]})
    fig = px.bar(cr, x="r", y="fitur", orientation="h", color="arah",
                 color_discrete_map={pos: STATUS_COLOR["Churn"], neg: STATUS_COLOR["Bertahan"]},
                 category_orders={"arah": [pos, neg]}, labels={"r": "Korelasi (r)", "fitur": "", "arah": ""})
    fig.update_traces(texttemplate="%{x:+.2f}", textposition="outside", cliponaxis=False,
                      hovertemplate="<b>%{y}</b><br>r = %{x:+.2f}<extra></extra>")
    m = max(r.abs().max() * 1.35, 0.1)
    fig.update_xaxes(range=[-m, m], zeroline=True, zerolinecolor=C["muted"])
    return fig_style(fig, legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))


def fig_corr_matrix(d):
    cols = [c for c in NUM_COLS if d[c].nunique() > 1] + (["churn"] if d.churn.nunique() > 1 else [])
    if len(cols) < 2:
        return empty_fig("Korelasi tidak bisa dihitung untuk filter ini")
    m = d[cols].corr()
    names = [LABELS[c] for c in cols]
    fig = go.Figure(go.Heatmap(
        z=m.values, x=names, y=names, text=m.values, texttemplate="%{text:.2f}", colorscale="RdBu",
        reversescale=True, zmin=-1, zmax=1, xgap=2, ygap=2, colorbar=dict(title="r", thickness=12),
        hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return fig_style(fig, height=380)


@callback(Output("eda-kpis", "children"), Output("eda-insights", "children"),
          Output("eda-heatmap", "figure"), Output("eda-trend", "figure"),
          Output("eda-corr", "figure"), Output("eda-corr-matrix", "figure"), *EDA_FILTERS)
def update_eda_overview(region, segment, contract, tenure):
    d = eda_filter(region, segment, contract, tenure)
    if d.empty:
        e = empty_fig(NO_DATA)
        return [kpi("Pelanggan", "0", "sesuai filter")], NO_DATA, e, e, e, e
    return eda_kpis(d), eda_insights(d), fig_heatmap(d), fig_trend(d), fig_corr(d), fig_corr_matrix(d)


@callback(Output("eda-cat-rate", "figure"), *EDA_FILTERS, Input("eda-cat", "value"))
def update_eda_categorical(region, segment, contract, tenure, col):
    d = eda_filter(region, segment, contract, tenure)
    if d.empty:
        return empty_fig(NO_DATA)
    g = churn_rates(d, col).sort_values("rate")
    fig = px.bar(g, x="rate", y=col, orientation="h", custom_data=["n", "churned", "rate"],
                 color_discrete_sequence=[C["accent"]], labels={"rate": "Churn rate", col: ""})
    fig.update_traces(texttemplate="%{x:.1%}", textposition="outside", cliponaxis=False,
                      hovertemplate=f"<b>%{{y}}</b><br>{RATE_HOVER}<extra></extra>")
    add_avg_line(fig, d.churn.mean(), horizontal=False)
    fig.update_xaxes(tickformat=".0%", range=[0, max(g.rate.max() * 1.25, 0.05)])
    return fig_style(fig)


@callback(Output("eda-num-dist", "figure"), Output("eda-num-rate", "figure"),
          *EDA_FILTERS, Input("eda-num", "value"))
def update_eda_numeric(region, segment, contract, tenure, col):
    d = eda_filter(region, segment, contract, tenure)
    if d.empty:
        return empty_fig(NO_DATA), empty_fig(NO_DATA)
    s = d.assign(status=d.churn.map({0: "Bertahan", 1: "Churn"})).dropna(subset=[col])
    discrete = col in ("num_complaints", "support_calls")
    dist = px.histogram(s, x=col, color="status", barmode="group", histnorm="percent", marginal="box",
                        nbins=None if discrete else 20, category_orders={"status": STATUS_ORDER},
                        color_discrete_map=STATUS_COLOR, labels={col: LABELS[col], "status": ""})
    if discrete:
        dist.update_traces(xbins=dict(size=1), selector=dict(type="histogram"))
    dist.update_traces(hovertemplate="%{x}<br>%{y:.1f}% pelanggan<extra>%{fullData.name}</extra>",
                       selector=dict(type="histogram"))
    dist.update_layout(yaxis_title="% pelanggan", bargap=0.1)
    fig_style(dist, legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))

    bins, names = NUM_BANDS[col]
    g = churn_rates(d.assign(band=pd.cut(d[col], bins=bins, labels=names)), "band")
    g["band"] = g.band.astype(str)
    rate = px.bar(g, x="band", y="rate", custom_data=["n", "churned", "rate"],
                  color_discrete_sequence=[C["accent"]], category_orders={"band": names},
                  labels={"band": LABELS[col], "rate": "Churn rate"})
    rate.update_traces(texttemplate="%{y:.0%}", textposition="outside", cliponaxis=False,
                       hovertemplate=f"<b>%{{x}}</b><br>{RATE_HOVER}<extra></extra>")
    add_avg_line(rate, d.churn.mean(), horizontal=True)
    rate.update_yaxes(tickformat=".0%", range=[0, max(g.rate.max() * 1.25, 0.05)])
    return dist, fig_style(rate)


@callback(Output("eda-region", "value"), Output("eda-segment", "value"),
          Output("eda-contract", "value"), Output("eda-tenure", "value"),
          Input("eda-reset", "n_clicks"), prevent_initial_call=True)
def reset_eda_filters(_):
    return None, None, None, [1, 72]

# --- Prediksi Single Observasi ---
@callback(Output("single-result", "children"), Input("single-btn", "n_clicks"),
          State("single-model", "value"), *[State(f"single-{c}", "value") for c in SINGLE_FIELDS],
          prevent_initial_call=True)
def run_single_prediction(_, model_key, *values):
    """Validasi input form → inference model terpilih → kartu hasil + gauge + rekomendasi."""
    row = dict(zip(SINGLE_FIELDS, values))
    kosong = [LABELS[c] for c, v in row.items() if v is None and c != "total_charges"]
    if kosong:
        return single_placeholder("Lengkapi isian berikut: " + ", ".join(kosong), "⚠️", RISK_COLOR["Tinggi"])
    if any(row[c] is not None and row[c] < 0 for c in NUM_COLS):
        return single_placeholder("Nilai numerik tidak boleh negatif.", "⚠️", RISK_COLOR["Tinggi"])
    if row["total_charges"] is None:                       # sama seperti imputasi di insight_bisnis.py
        row["total_charges"] = row["monthly_charges"] * row["tenure_months"]

    if not model_key or not model_available(model_key):
        return single_placeholder("File model belum tersedia di folder models/. Prediksi belum bisa "
                                  "dijalankan.", "📦", RISK_COLOR["Tinggi"])
    try:
        model, meta = get_model(model_key)
        df = pd.DataFrame([row])
        errors = validate_input(df, meta)
        if errors:
            return single_placeholder(" · ".join(errors), "⚠️", RISK_COLOR["Tinggi"])
        hasil = predict(model, df, meta).iloc[0]
    except Exception as e:                                  # mis. file model rusak / versi tidak cocok
        return single_placeholder(f"Gagal menjalankan model: {e}", "❌", RISK_COLOR["Tinggi"])

    p, risiko = float(hasil["churn_probability"]), hasil["risk_category"]
    kelas = "Churn" if hasil["predicted_churn"] == 1 else "Bertahan"
    judul_aksi, aksi = AKSI_RISIKO.get(risiko, ("-", "-"))
    warna = RISK_COLOR.get(risiko, C["ink"])
    rentang = " · ".join(f"{k} {lo:.0%}–{hi:.0%}" for k, (lo, hi) in meta["risk_bins"].items())

    def tile(judul, nilai, sub, color):
        return html.Div(style={**CARD, "flex": "1 1 160px", "borderTop": f"4px solid {color}"}, children=[
            html.Div(judul, style={"color": C["muted"], "fontSize": "13px"}),
            html.Div(nilai, style={"fontSize": "26px", "fontWeight": 700, "color": color}),
            html.Div(sub, style={"color": C["muted"], "fontSize": "12px"})])

    info_model = [MODEL_REGISTRY[model_key]["label"]]
    if "holdout_metrics" in meta and "roc_auc" in meta["holdout_metrics"]:
        info_model.append(f"ROC-AUC holdout {meta['holdout_metrics']['roc_auc']:.3f}")
    saran = faktor_risiko(row)
    return html.Div([
        html.Div(style=ROW, children=[
            tile("Probabilitas churn", f"{p:.1%}", f"rata-rata churn historis {RAW.churn.mean():.1%}", warna),
            tile("Prediksi kelas", kelas, f"threshold {meta['decision_threshold']:.0%}", STATUS_COLOR[kelas]),
            tile("Kategori risiko", risiko, rentang, warna)]),
        card(dcc.Graph(figure=gauge_fig(p, meta), config={"displayModeBar": False}), marginBottom="12px"),
        card(html.H4("Rekomendasi tindakan", style={"marginTop": 0}),
             html.Div(style={"borderLeft": f"4px solid {warna}", "background": warna + "14",
                             "padding": "10px 14px", "borderRadius": "6px"}, children=[
                 html.Div(judul_aksi, style={"fontWeight": 700, "color": warna}),
                 html.Div(aksi, style={"marginTop": "4px"})]),
             *([html.Div("Faktor yang perlu diperhatikan pada pelanggan ini:",
                         style={"marginTop": "12px", "fontWeight": 600}),
                html.Ul([html.Li(s) for s in saran], style={"margin": "6px 0 0", "paddingLeft": "20px"})]
               if saran else []),
             html.Div(f"Model: {' · '.join(info_model)} · Total tagihan dipakai: {rupiah(row['total_charges'])}",
                      style={"color": C["muted"], "fontSize": "12px", "marginTop": "12px"})),
    ])


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
