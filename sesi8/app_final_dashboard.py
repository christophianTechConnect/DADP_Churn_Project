# sesi8/app_final_dashboard.py — FINAL DASHBOARD PROJECT (Dash + Plotly + callback + ML inference)
#
# STATUS: kerangka navigasi (sidebar pop-up) saja. Isi tiap halaman BELUM dibuat —
# lihat README.md bagian "Pembagian tugas" untuk requirement tiap halaman.
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
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table, Input, Output, State, callback, ctx, no_update

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from utils.model_utils import load_model, load_metadata, validate_input, predict  # noqa: E402

# ============ 0a. DAFTAR MODEL (dipakai dropdown "Pilih model" di halaman prediksi) ============
# Menambah model baru cukup dengan menambah satu entri di sini; layout halaman tidak perlu diubah.
MODEL_REGISTRY = {
    "gb_v1": {"label": "Gradient Boosting v1.0.0",
              "model": ROOT / "models" / "churn_model.joblib",
              "meta": ROOT / "models" / "model_metadata.json"},
    # "lr_v1": {"label": "Logistic Regression v1.0.0", "model": ROOT / "models" / "...joblib",
    #           "meta": ROOT / "models" / "...json"},
}
_MODEL_CACHE = {}


def model_available(key):
    cfg = MODEL_REGISTRY[key]
    return cfg["model"].exists() and cfg["meta"].exists()


def model_options():
    """Opsi dropdown model; model yang file-nya belum ada tetap tampil tapi tidak bisa dipilih."""
    return [{"label": cfg["label"] + ("" if model_available(k) else "  (file model belum tersedia)"),
             "value": k, "disabled": not model_available(k)} for k, cfg in MODEL_REGISTRY.items()]


def default_model():
    return next((k for k in MODEL_REGISTRY if model_available(k)), next(iter(MODEL_REGISTRY)))


def get_model(key):
    """Memuat model + metadata sekali, lalu disimpan di cache untuk prediksi berikutnya."""
    if key not in _MODEL_CACHE:
        cfg = MODEL_REGISTRY[key]
        _MODEL_CACHE[key] = (load_model(cfg["model"]), load_metadata(cfg["meta"]))
    return _MODEL_CACHE[key]

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
         ("multi", "📁  Prediksi Multi Observasi")]
PAGE_TITLE = {k: t.split("  ", 1)[1] for k, t in PAGES}


# ============ 3. ISI HALAMAN (placeholder — dikerjakan oleh anggota tim) ============
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
    # TODO: upload file CSV/Excel → prediksi seluruh baris; grafik & tabel HANYA dari file — lihat README.
    # CATATAN MODEL: sama seperti Prediksi Single — sediakan dropdown "Pilih model".
    # Saat ini baru ada SATU model (Gradient Boosting v1.0.0).
    return placeholder("multi", "Upload file pelanggan; hasil prediksi hanya menampilkan data dari file. "
                                "Bisa memilih model (saat ini baru 1 model tersedia).")


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


if __name__ == "__main__":
    app.run(debug=True, port=8053)
