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
from dash import Dash, html, dcc, Input, Output, callback, ctx, no_update

# ============ 1. STYLE ============
C = {"bg": "#F5F7FB", "card": "white", "ink": "#1F2A44", "muted": "#6B7280", "accent": "#2F5DA8"}
CARD = {"background": C["card"], "borderRadius": "12px", "padding": "16px",
        "boxShadow": "0 1px 3px rgba(0,0,0,.08)"}
BTN = {"background": C["accent"], "color": "white", "border": 0, "padding": "6px 14px",
       "borderRadius": "8px", "cursor": "pointer", "fontSize": "20px"}

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
    # TODO: profil dataset historis (data/customer_churn.csv) — lihat README.
    return placeholder("overview", "Profil dataset: ukuran data, kamus data, missing value, statistik.")


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


if __name__ == "__main__":
    app.run(debug=True, port=8053)
