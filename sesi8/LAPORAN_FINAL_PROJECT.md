# Final Project — Customer Churn Intelligence Dashboard

Dokumen pendamping dashboard Sesi 8 (Buku Praktikum *Interactive Dashboard Development with Streamlit & Dash*, ITSB Corporation).

- **Aplikasi:** `sesi8/app_final_dashboard.py` (Dash + Plotly + callback + inference model final)
- **Cara menjalankan:** `python sesi8/app_final_dashboard.py` → buka http://127.0.0.1:8053
- **Angka insight:** dihitung ulang oleh `python sesi8/insight_bisnis.py`

---

## 1. Business Problem
Sebuah penyedia layanan internet kehilangan 34,9% pelanggannya (churn). Pertanyaan bisnis yang dijawab dashboard:

> **Pelanggan mana yang berisiko churn, di segmen mana risiko terkonsentrasi, dan tindakan apa yang harus dilakukan?**

## 2. Data Understanding
| Aspek | Keterangan |
|---|---|
| Sumber | `data/customer_churn.csv`: data latihan sintetis, seed tetap |
| Ukuran | 3.000 pelanggan × 13 kolom; pendaftaran Januari 2023 – Desember 2025 |
| Target | `churn` (1 = berhenti berlangganan); churn rate 34,9% (1.048 pelanggan) |
| Fitur | 5 numerik (tenure, tagihan bulanan, total tagihan, keluhan, panggilan CS) dan 5 kategorik (region, segmen, kontrak, metode pembayaran, layanan internet) |
| Kualitas data | Tidak ada duplikat `customer_id`. 45 baris `total_charges` kosong; di dashboard final diisi `monthly_charges × tenure_months` |
| Data inference | `data/new_customers.csv`: 200 pelanggan baru tanpa label |

## 3. EDA
Dikerjakan di Sesi 2 (`sesi2/eda_churn.py`, 16 grafik; `sesi2/app_eda.py`, EDA interaktif 7 tab). Temuan awal:
- **Jenis kontrak adalah pembeda terkuat:** churn Bulanan 54,6%, Tahunan 16,8%, Dua Tahun 9,7%.
- **Masa awal paling rawan:** churn 54,6% pada tenure 0–12 bulan, 6,7% pada tenure di atas 48 bulan.
- **Keluhan berulang adalah sinyal dini:** pelanggan Bulanan dengan ≥ 3 keluhan memiliki churn 71,9%.
- **Region bukan pembeda utama:** 31,4% (Bali-Nusa Tenggara) sampai 37,3% (Kalimantan). Namun kombinasi Korporat–Sulawesi mencapai 52,6%.

## 4. Visualization
Enam chart Plotly di tab *Overview & Risiko*. Setiap chart menjawab satu pertanyaan:

| Chart | Pertanyaan | Alasan pemilihan |
|---|---|---|
| Donut komposisi risiko | Berapa porsi pelanggan berisiko? | Komposisi 3 kategori; total di tengah |
| Bar horizontal risiko per region | Region mana yang diprioritaskan? | Perbandingan, terurut, label panjang |
| Line aktual vs prediksi per kuartal | Apakah model sejalan dengan kenyataan dari waktu ke waktu? | Tren berurutan |
| Stacked bar risiko per kontrak | Kontrak mana yang menyumbang risiko? | Komposisi di dalam kategori |
| Histogram probabilitas + garis threshold | Bagaimana sebaran skor model? | Distribusi + referensi keputusan |
| Scatter aktual vs prediksi per region | Seberapa akurat prediksi tiap region? | Hubungan dua ukuran; ukuran titik = jumlah pelanggan |

## 5. Final ML Model
Model **tidak dilatih ulang** di pelatihan ini; file disediakan fasilitator.

| Atribut | Nilai |
|---|---|
| File | `models/churn_model.joblib` (+ kontrak input `models/model_metadata.json`) |
| Jenis | scikit-learn `Pipeline`: imputasi median + standarisasi (numerik), one-hot (kategorik) → `GradientBoostingClassifier` |
| Versi | Customer Churn Classifier v1.0.0, scikit-learn 1.9.1 |
| Kinerja holdout | ROC-AUC 0,789 · akurasi 0,718 · F1 0,579 |

## 6. Model Loading & Inference
Logika inference ada di `utils/model_utils.py` dan dipakai ulang oleh Streamlit (Sesi 4) maupun Dash (Sesi 8):
1. `load_model()` / `load_metadata()`: model dan kontrak dimuat **sekali** saat aplikasi start.
2. `validate_input()`: menolak kolom hilang, kategori tidak dikenal (misalnya "Mingguan"), atau nilai non-numerik.
3. `prepare_input()`: memilih dan mengurutkan kolom sesuai urutan saat model difinalisasi.
4. `predict()`: `predict_proba` → probabilitas, kelas (threshold 0,5), dan kategori risiko.

Di dashboard final, **batch inference** dijalankan untuk 3.000 pelanggan saat start, sehingga callback filter hanya memotong DataFrame yang sudah berisi prediksi.

## 7. Prediction Output
| Output | Bentuk di dashboard |
|---|---|
| Probabilitas churn (0–1) | KPI rata-rata, histogram, tabel prioritas, angka besar di simulasi |
| Kelas prediksi | "Churn" / "Bertahan" (threshold 0,5) di simulasi |
| Kategori risiko | Rendah < 0,30 · Sedang 0,30–0,60 · Tinggi ≥ 0,60; dipakai sebagai **filter** dan warna di semua chart |
| Rekomendasi | Tindakan per kategori risiko di tab simulasi |

## 8. Interactive Components
- **Sidebar filter (5):** region, segmen, jenis kontrak, kategori risiko (hasil prediksi), tenure.
- **Form simulasi (9 input):** region, segmen, kontrak, pembayaran, layanan, tenure, tagihan, keluhan, panggilan CS, plus tombol *Jalankan prediksi*.
- **Tabel prioritas retensi:** sorting, filter per kolom, dan paging bawaan DataTable.
- **Interaksi Plotly bawaan:** hover, zoom, dan legend toggle.

## 9. Callback / App Logic
| Callback | Input | Output |
|---|---|---|
| `update_overview` | 5 filter sidebar | 4 KPI, 6 chart, data tabel, judul tabel |
| `run_prediction` | tombol (Input) + 9 field form (State) | hasil prediksi + gauge |

Form simulasi memakai `State`, sehingga model hanya dipanggil saat tombol diklik, bukan setiap kali angka diketik. Input kosong dihentikan dengan `PreventUpdate`. Filter yang menghasilkan data kosong menampilkan pesan *"Tidak ada data — longgarkan filter"* tanpa error.

## 10. Dashboard
Layout: header (judul + pertanyaan bisnis + periode data) → sidebar filter → tab *Overview & Risiko* (4 KPI → 6 chart → tabel) dan tab *Simulasi Prediksi*.

**Pemenuhan Dashboard Minimum Requirements (buku 8.5)**

| Requirement | Implementasi | Status |
|---|---|---|
| Minimal 4 KPI cards | Pelanggan, churn rate aktual, rata-rata prob. churn, risiko tinggi (+ tagihan berisiko) | ✅ |
| Minimal 3 visualisasi interaktif | 6 chart Plotly + gauge | ✅ |
| Minimal 2 filter/input | 5 filter + 9 input simulasi | ✅ |
| Minimal 1 tampilan prediction output | Filter risiko, histogram probabilitas, tabel prioritas, gauge | ✅ |
| Minimal 1 tabel detail | DataTable prioritas retensi (sort, filter, paging) | ✅ |
| Layout jelas | Header → KPI → chart → tabel; simulasi di tab terpisah | ✅ |
| Judul, unit, periode, label jelas | Judul deskriptif, format %, Rupiah, periode 2023–2025 | ✅ |
| Menjawab pertanyaan bisnis | Pertanyaan tertulis di header; dijawab KPI risiko dan tabel prioritas | ✅ |

**Checkpoint buku 8.6 (sudah diuji):**
1. Tanpa filter: 3.000 pelanggan · churn aktual 34,9% · rata-rata probabilitas 34,7% · 624 risiko tinggi (Rp288.289.000/bln) ✅
2. Hanya kategori Tinggi: semua chart dan tabel hanya berisi pelanggan risiko tinggi ✅
3. Semua jenis kontrak dikosongkan: muncul "Tidak ada data — longgarkan filter" tanpa error ✅
4. Simulasi dengan nilai default: 71,1% · Prediksi Churn · Risiko Tinggi · gauge merah ✅

## 11. Insight & Recommendation
| # | Temuan (angka) | Rekomendasi |
|---|---|---|
| 1 | **624 pelanggan (20,8%)** berisiko tinggi, dengan total tagihan **Rp288,3 juta/bulan** | Jadikan kelompok ini fokus program retensi bulan ini |
| 2 | **96,0%** pelanggan risiko tinggi berkontrak **Bulanan** | Migrasi ke kontrak Tahunan dengan insentif diskon |
| 3 | Rata-rata tenure risiko tinggi **14,0 bulan**, vs 26,5 bulan untuk seluruh pelanggan | Perkuat onboarding dan kontak proaktif pada 12 bulan pertama |
| 4 | Rata-rata keluhan risiko tinggi **1,78**, vs 1,13 untuk seluruh pelanggan | SLA penyelesaian keluhan maksimal 48 jam untuk pelanggan berisiko tinggi |
| 5 | Risiko tinggi di **Transfer Bank 27,4%**, E-Wallet 19,6%, Kartu Kredit 17,0%, Virtual Account 16,6% | Dorong autodebit atau e-wallet untuk mengurangi friksi pembayaran |
| 6 | **Sumatera** memiliki porsi risiko tinggi terbesar (**24,3%**); Bali-Nusa Tenggara terendah (18,3%) | Prioritaskan kampanye retensi regional di Sumatera |
| 7 | Kategori risiko konsisten dengan data historis: churn aktual **78,5%** (Tinggi), 45,4% (Sedang), 9,7% (Rendah) | Gunakan metrik holdout (ROC-AUC 0,789) sebagai acuan kinerja yang adil, karena sebagian data ini dipakai saat training |
| 8 | **134 pelanggan aktif** (belum churn) masuk kategori risiko tinggi | Jadikan daftar kerja pertama tim retensi minggu ini → `sesi8/daftar_kerja_retensi.csv` |

**Keterbatasan:**
- Data sintetis untuk latihan.
- Angka di nomor 7 sebagian memakai data training, sehingga cenderung lebih optimis daripada kinerja pada pelanggan baru.
- Prediksi adalah perkiraan risiko, bukan kepastian.
- Threshold 0,5 dan batas kategori risiko adalah keputusan bisnis yang dapat disesuaikan (lihat slider threshold di Sesi 4).

---

## Struktur presentasi 5–7 menit (buku 8.8)
1. **Masalah bisnis (30 detik):** churn 34,9%. Pelanggan mana yang berisiko, dan apa yang harus dilakukan?
2. **Data dan model (45 detik):** 3.000 pelanggan 2023–2025; GradientBoosting final (ROC-AUC 0,789). Probabilitas dipetakan ke kategori Rendah/Sedang/Tinggi.
3. **Tur dashboard (2 menit):** 4 KPI → donut risiko → bar region → tren aktual vs prediksi → kontrak → histogram threshold → tabel prioritas.
4. **Demo interaksi (1 menit):** centang hanya *Tinggi* di filter risiko; lalu tab Simulasi dengan nilai default (71,1%, Tinggi), ubah kontrak ke Dua Tahun dan tenure ke 48, dan lihat risiko turun.
5. **Insight (1,5 menit):** insight nomor 1, 2, 3, dan 8.
6. **Rekomendasi dan keterbatasan (1 menit):** migrasi kontrak, onboarding 12 bulan, SLA keluhan, daftar 134 pelanggan; data sintetis dan threshold yang dapat disesuaikan.

## Checklist final submission (buku 8.10)
| Item | Lokasi |
|---|---|
| ☑ Source code Streamlit/Dash | `sesi1/` – `sesi8/` |
| ☑ Dataset / analytical data | `data/customer_churn.csv`, `data/new_customers.csv` |
| ☑ EDA workflow | `sesi2/eda_churn.py`, `sesi2/app_eda.py` |
| ☑ Interactive Plotly visualizations | `sesi6/plotly_galeri.py`, `sesi8/app_final_dashboard.py` |
| ☑ Final trained / selected best-performing ML model | `models/churn_model.joblib`, `models/model_metadata.json` |
| ☑ Model loading & inference workflow | `utils/model_utils.py`, `sesi4/cek_inference.py` |
| ☑ Prediction result/output | `sesi4/app_prediksi.py`, tab Simulasi di dashboard final, `sesi8/daftar_kerja_retensi.csv` |
| ☑ Interactive dashboard | `sesi8/app_final_dashboard.py` |
| ☑ Filter/input components | Sidebar filter + form simulasi |
| ☑ Callback implementation (Dash) | `update_overview`, `run_prediction`; juga `sesi7/` |
| ☑ Business insight | Bagian 11 dokumen ini; `sesi8/insight_bisnis.py` |
| ☑ Recommendation | Bagian 11 dokumen ini |
| ☐ Presentasi final project | Mengikuti struktur di atas (dilakukan langsung) |
