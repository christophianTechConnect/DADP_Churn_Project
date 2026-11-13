# Final Project — Customer Churn Intelligence Dashboard

Dokumen pendamping dashboard final project (Buku Praktikum *Interactive Dashboard Development with Streamlit & Dash*, ITSB Corporation).

- **Aplikasi:** `sesi8/app_final_dashboard.py` (Dash + Plotly + callback + inference 3 model)
- **Cara menjalankan:** `pip install -r requirements.txt`, lalu `python sesi8/app_final_dashboard.py` → buka http://127.0.0.1:8053
- **Angka insight:** dihitung ulang oleh `python sesi8/insight_bisnis.py`
- **Tes:** `python -m unittest discover -s tests -v`
- **Tim:** daftar kontributor ada di halaman **Team** pada dashboard (diambil dari riwayat git)

---

## 1. Business Problem
Sebuah penyedia layanan internet kehilangan 34,9% pelanggannya (churn). Pertanyaan bisnis yang dijawab dashboard:

> **Pelanggan mana yang berisiko churn, di segmen mana risiko terkonsentrasi, dan tindakan apa yang harus dilakukan?**

## 2. Data Understanding
| Aspek | Keterangan |
|---|---|
| Sumber | `data/customer_churn.csv`: data latihan sintetis |
| Ukuran | 3.000 pelanggan × 13 kolom; pendaftaran Januari 2023 – Desember 2025 |
| Target | `churn` (1 = berhenti berlangganan); churn rate 34,9% (1.048 pelanggan) |
| Fitur | 5 numerik (tenure, tagihan bulanan, total tagihan, keluhan, panggilan CS) dan 5 kategorikal (region, segmen, kontrak, metode pembayaran, layanan internet) |
| Kualitas data | Tidak ada duplikat `customer_id`. 45 baris `total_charges` kosong (ditampilkan apa adanya di Data Overview; di prediksi Single diisi `monthly_charges × tenure_months` jika dikosongkan; model juga mengimputasi median) |
| Data inference | `data/new_customers.csv`: 200 pelanggan baru tanpa label, contoh file untuk halaman Prediksi Multi |

Profil lengkap (KPI, kamus data, missing value, statistik deskriptif, pratinjau data) ada di halaman **Data Overview**.

## 3. EDA
Dikerjakan di halaman **EDA** dashboard. Semua KPI, grafik, dan insight dihitung ulang mengikuti filter (region, segmen, jenis kontrak, rentang tenure). Temuan pada data lengkap:
- **Jenis kontrak adalah pembeda terkuat:** churn Bulanan 54,6%, Tahunan 16,8%, Dua Tahun 9,7%.
- **Masa awal paling rawan:** churn 54,6% pada tenure 1–12 bulan, turun ke 6,7% pada tenure di atas 48 bulan. Tenure adalah fitur numerik dengan korelasi terkuat terhadap churn (r = −0,33).
- **Keluhan berulang adalah sinyal dini:** pelanggan Bulanan dengan ≥ 3 keluhan memiliki churn 71,9% (korelasi keluhan dengan churn r = 0,18; panggilan CS r = 0,14).
- **Region bukan pembeda utama:** 31,4% (Bali-Nusa Tenggara) sampai 37,3% (Kalimantan). Namun kombinasi Korporat–Sulawesi mencapai 52,6%.

## 4. Visualization
Dashboard terdiri dari 5 halaman (navigasi lewat sidebar pop-up ☰):

| Halaman | Visualisasi | Pertanyaan yang dijawab |
|---|---|---|
| Data Overview | Bar distribusi target, bar missing value per kolom, bar komposisi fitur kategorikal (facet) | Seperti apa data dan kualitasnya? |
| EDA | Bar churn rate per fitur kategorikal (+ garis rata-rata) | Kategori mana yang paling banyak churn? |
| | Heatmap churn rate segmen × jenis kontrak | Kombinasi mana yang paling rawan? |
| | Histogram + box plot fitur numerik per status churn; bar churn rate per kelompok nilai | Bagaimana nilai numerik berbeda antara churn dan bertahan? |
| | Line tren churn rate per kuartal pendaftaran | Apakah churn berubah dari waktu ke waktu? |
| | Bar korelasi fitur dengan churn; heatmap matriks korelasi | Fitur numerik mana yang paling terkait dengan churn? |
| Prediksi Single | Gauge probabilitas (zona warna kategori risiko + garis threshold) | Seberapa besar risiko pelanggan ini? |
| Prediksi Multi | Donut komposisi risiko; stacked bar pelanggan per region dan risiko | Berapa banyak pelanggan dalam file yang berisiko, dan di mana? |
| Team | Kartu kontributor | Siapa yang mengerjakan proyek ini? |

## 5. Final ML Model
Tiga model terbaik berdasarkan **ROC-AUC cross-validation 5-fold stratified** (split train/test 80/20, stratified, `random_state=42`) tersedia di dropdown kedua halaman prediksi. Daftarnya dibaca dari `models/top_models_registry.json`; model #1 terpilih secara default.

| Rank | Model | File | CV ROC-AUC | Holdout ROC-AUC | Holdout recall | Holdout F1 |
|---|---|---|---|---|---|---|
| 1 | Logistic Regression | `churn_model_logreg.joblib` | 0,800 ± 0,013 | 0,796 | 0,524 | 0,569 |
| 2 | Logistic Regression + RandomOverSampler | `churn_model_logreg_ros.joblib` | 0,799 ± 0,013 | 0,794 | 0,719 | 0,632 |
| 3 | Gradient Boosting (model final asli fasilitator) | `churn_model.joblib` | 0,792 ± 0,012 | 0,789 | 0,552 | 0,579 |

- Semua model berupa `Pipeline` (imputasi median + standarisasi numerik, one-hot kategorikal → classifier), scikit-learn 1.9.1. Model #2 memakai pipeline `imbalanced-learn` dengan oversampling hanya pada data latih.
- ROC-AUC ketiga model hampir sama. Model #2 menangkap churn paling banyak (recall tertinggi) dengan konsekuensi lebih banyak *false alarm*, sehingga probabilitasnya cenderung lebih tinggi.
- Model **tidak dilatih ulang** di dashboard; file `.joblib` hanya dimuat.

## 6. Model Loading & Inference
- `utils/multi_predictions.py`: `MODEL_REGISTRY` dibangun dari `top_models_registry.json`; `get_model_bundle()` memuat model + metadata **sekali** lalu disimpan di cache. Metrik holdout yang ditampilkan diambil dari entri model masing-masing.
- `utils/model_utils.py`:
  1. `validate_input()`: menolak kolom hilang, kategori tidak dikenal (misalnya "Mingguan"), atau nilai non-numerik.
  2. `prepare_input()`: memilih dan mengurutkan kolom sesuai `feature_order`.
  3. `predict()`: `predict_proba` → probabilitas, kelas (threshold 0,5), dan kategori risiko.
- Ketiga model memakai kontrak input yang sama: `models/model_metadata.json` (fitur, kategori valid, threshold, risk bins).
- `predict_upload()` (halaman Multi) menambahkan validasi file: hanya `.csv`, file tidak boleh kosong/rusak, kolom numerik harus angka terhingga, kolom kategorikal tidak boleh kosong.

## 7. Prediction Output
| Output | Bentuk di dashboard |
|---|---|
| Probabilitas churn (0–1) | Angka besar + gauge (Single); KPI rata-rata + kolom tabel (Multi) |
| Kelas prediksi | "Churn" / "Bertahan" pada threshold 0,5 |
| Kategori risiko | Rendah < 0,30 · Sedang 0,30–0,60 · Tinggi ≥ 0,60; warna di gauge, donut, dan bar |
| Rekomendasi | Tindakan per kategori risiko + saran spesifik dari karakteristik pelanggan (kontrak Bulanan, tenure < 12 bulan, ≥ 2 keluhan, ≥ 5 panggilan CS, Transfer Bank) |
| File hasil | Tombol **Unduh hasil CSV** di halaman Multi (semua kolom upload + probabilitas, kelas, kategori risiko) |

Contoh hasil pada `data/new_customers.csv` (200 pelanggan):

| Model | Prediksi churn | Risiko Tinggi | Rata-rata probabilitas |
|---|---|---|---|
| #1 Logistic Regression | 61 | 45 | 34,1% |
| #2 LR + Oversampling | 92 | 71 | 44,1% |
| #3 Gradient Boosting | 65 | 51 | 34,8% |

## 8. Interactive Components
- **Sidebar pop-up (☰):** navigasi 5 halaman; tertutup lewat ✕, klik di luar sidebar, atau memilih menu.
- **EDA:** 3 dropdown multi-pilih (region, segmen, jenis kontrak), range slider tenure, tombol *Reset filter*, dropdown pilihan fitur kategorikal dan fitur numerik.
- **Prediksi Single:** dropdown model + form 10 input (5 dropdown kategorikal, 5 input angka; total tagihan opsional) + tombol *Prediksi churn*.
- **Prediksi Multi:** dropdown model, upload CSV (drag & drop), tombol unduh hasil. Mengganti model menjalankan ulang prediksi pada file yang sama.
- **Tabel:** DataTable dengan sorting, filter per kolom, dan paging (pratinjau data mentah, hasil prediksi Multi).
- **Interaksi Plotly bawaan:** hover, zoom, legend toggle.

## 9. Callback / App Logic
| Callback | Input | Output |
|---|---|---|
| `toggle_drawer` | tombol ☰, ✕, backdrop, 5 menu | status sidebar, halaman aktif |
| `show_page` | halaman aktif | tampil/sembunyi 5 halaman, menu aktif, subjudul |
| `update_eda_overview` | 4 filter | KPI, insight, heatmap, tren, korelasi, matriks korelasi |
| `update_eda_categorical` | 4 filter + fitur kategorikal | bar churn rate |
| `update_eda_numeric` | 4 filter + fitur numerik | distribusi + churn rate per kelompok |
| `reset_eda_filters` | tombol reset | 4 filter |
| `run_single_prediction` | tombol (Input) + model + 10 field (State) | kartu hasil, gauge, rekomendasi |
| `update_multi` | file upload + model (Input), nama file (State) | pesan validasi, KPI, grafik, tabel, data hasil, status tombol unduh |
| `download_multi` | tombol unduh + data hasil (State) | file CSV |

- Form Single memakai `State`, sehingga model hanya dipanggil saat tombol diklik. Input kosong/negatif, model belum tersedia, atau error model ditampilkan sebagai pesan, bukan crash.
- Halaman Multi kosong sebelum ada upload; upload yang tidak valid menampilkan daftar error dan **menghapus** hasil sebelumnya.
- Filter EDA yang menghasilkan data kosong menampilkan pesan *"Tidak ada data"* tanpa error.
- Semua halaman dirender sekali dan hanya disembunyikan/ditampilkan, sehingga isian form, filter, dan file upload tidak hilang saat berpindah halaman.

## 10. Dashboard
Layout: header (judul + subjudul halaman aktif) → tombol ☰ → isi halaman. Setiap halaman: judul + deskripsi → KPI → grafik → tabel.

**Pemenuhan Dashboard Minimum Requirements (buku 8.5)**

| Requirement | Implementasi | Status |
|---|---|---|
| Minimal 4 KPI cards | Data Overview (6), EDA (5), Prediksi Multi (4), Team (3) | ✅ |
| Minimal 3 visualisasi interaktif | 3 di Data Overview, 7 di EDA, gauge di Single, 2 di Multi | ✅ |
| Minimal 2 filter/input | 4 filter + 2 pemilih fitur (EDA), 10 input + model (Single), upload + model (Multi) | ✅ |
| Minimal 1 tampilan prediction output | Gauge + kartu hasil (Single), KPI + donut + tabel hasil (Multi) | ✅ |
| Minimal 1 tabel detail | Kamus data, statistik deskriptif, pratinjau data, tabel hasil prediksi | ✅ |
| Layout jelas | Navigasi per halaman; urutan KPI → grafik → tabel | ✅ |
| Judul, unit, periode, label jelas | Judul deskriptif, format %, Rupiah, periode pendaftaran | ✅ |
| Menjawab pertanyaan bisnis | Kategori risiko + rekomendasi tindakan per pelanggan; insight di EDA | ✅ |

**Pengujian** (`tests/test_multi_predictions.py`, 8 tes, lulus): halaman Multi kosong sebelum upload; hanya pelanggan dari file yang ditampilkan dan diunduh; kategori tidak dikenal, kolom hilang, file kosong/rusak/bukan CSV, angka tidak valid ditolak; ketiga model memprediksi dengan metrik masing-masing; layout dan endpoint callback Dash merespons.

## 11. Insight & Recommendation
Angka di bawah dihitung oleh `sesi8/insight_bisnis.py` dengan model **#3 Gradient Boosting** (model final asli) pada 3.000 pelanggan historis.

| # | Temuan (angka) | Rekomendasi |
|---|---|---|
| 1 | **624 pelanggan (20,8%)** berisiko tinggi, dengan total tagihan **Rp288,3 juta/bulan** | Jadikan kelompok ini fokus program retensi bulan ini |
| 2 | **96,0%** pelanggan risiko tinggi berkontrak **Bulanan** | Migrasi ke kontrak Tahunan dengan insentif diskon |
| 3 | Rata-rata tenure risiko tinggi **14,0 bulan**, vs 26,5 bulan untuk seluruh pelanggan | Perkuat onboarding dan kontak proaktif pada 12 bulan pertama |
| 4 | Rata-rata keluhan risiko tinggi **1,78**, vs 1,13 untuk seluruh pelanggan | SLA penyelesaian keluhan maksimal 48 jam untuk pelanggan berisiko tinggi |
| 5 | Risiko tinggi di **Transfer Bank 27,4%**, E-Wallet 19,6%, Kartu Kredit 17,0%, Virtual Account 16,6% | Dorong autodebit atau e-wallet untuk mengurangi friksi pembayaran |
| 6 | **Sumatera** memiliki porsi risiko tinggi terbesar (**24,3%**); Bali-Nusa Tenggara terendah (18,3%) | Prioritaskan kampanye retensi regional di Sumatera |
| 7 | Kategori risiko konsisten dengan data historis: churn aktual **78,5%** (Tinggi), 45,4% (Sedang), 9,7% (Rendah) | Gunakan metrik holdout/CV sebagai acuan kinerja yang adil, karena sebagian data ini dipakai saat training |
| 8 | **134 pelanggan aktif** (belum churn) masuk kategori risiko tinggi | Jadikan daftar kerja pertama tim retensi minggu ini → `sesi8/daftar_kerja_retensi.csv` |

**Keterbatasan:**
- Data sintetis untuk latihan.
- Angka nomor 7 sebagian memakai data training, sehingga cenderung lebih optimis daripada kinerja pada pelanggan baru.
- Prediksi adalah perkiraan risiko, bukan kepastian. Ketiga model memberi probabilitas berbeda untuk pelanggan yang sama (form default Single: 89,3% / 93,5% / 86,0%), walaupun kategori risikonya sama.
- Threshold 0,5 dan batas kategori risiko adalah keputusan bisnis yang dapat disesuaikan di `models/model_metadata.json`.
- Upload Multi hanya menerima CSV (Excel opsional di tugas).

---

## Struktur presentasi 5–7 menit (buku 8.8)
1. **Masalah bisnis (30 detik):** churn 34,9%. Pelanggan mana yang berisiko, dan apa yang harus dilakukan?
2. **Data dan model (1 menit):** 3.000 pelanggan 2023–2025 (Data Overview); 3 model terbaik berdasarkan CV ROC-AUC (≈0,79–0,80). Probabilitas dipetakan ke kategori Rendah/Sedang/Tinggi.
3. **EDA (1,5 menit):** filter kontrak Bulanan → churn rate naik; heatmap segmen × kontrak; tenure dan keluhan per status churn; insight utama.
4. **Demo prediksi (1,5 menit):** Single dengan nilai default (Risiko Tinggi, rekomendasi muncul), ubah kontrak ke Dua Tahun dan tenure ke 48 dan lihat risiko turun; ganti model di dropdown. Multi: upload `data/new_customers.csv`, tunjukkan KPI, donut, tabel, unduh CSV, lalu upload file dengan kategori "Mingguan" untuk menunjukkan validasi.
5. **Insight dan rekomendasi (1 menit):** insight nomor 1, 2, 3, dan 8.
6. **Keterbatasan dan tim (30 detik):** data sintetis, threshold dapat disesuaikan; halaman Team.

## Checklist final submission (buku 8.10)
| Item | Lokasi |
|---|---|
| ☑ Source code Dash | `sesi8/app_final_dashboard.py`, `utils/` |
| ☑ Dataset / analytical data | `data/customer_churn.csv`, `data/new_customers.csv` |
| ☑ EDA workflow | Halaman EDA (`page_eda()` + callback `update_eda_*`) |
| ☑ Interactive Plotly visualizations | Halaman Data Overview, EDA, Prediksi Single, Prediksi Multi |
| ☑ Final trained / selected best-performing ML model | `models/` (3 model `.joblib`, `model_metadata.json`, `top_models_registry.json`) |
| ☑ Model loading & inference workflow | `utils/model_utils.py`, `utils/multi_predictions.py` |
| ☑ Prediction result/output | Halaman Prediksi Single & Multi, `sesi8/daftar_kerja_retensi.csv` |
| ☑ Interactive dashboard | `sesi8/app_final_dashboard.py` |
| ☑ Filter/input components | Filter EDA, form Single, upload + dropdown model |
| ☑ Callback implementation (Dash) | Bagian 9 dokumen ini |
| ☑ Pengujian | `tests/test_multi_predictions.py` |
| ☑ Business insight | Bagian 3 dan 11 dokumen ini; `sesi8/insight_bisnis.py` |
| ☑ Recommendation | Bagian 11 dokumen ini; rekomendasi di halaman Prediksi Single |
| ☐ Presentasi final project | Mengikuti struktur di atas (dilakukan langsung) |
