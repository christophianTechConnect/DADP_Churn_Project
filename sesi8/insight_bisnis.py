# sesi8/insight_bisnis.py — menghitung angka untuk insight & rekomendasi (buku 8.7)
# Jalankan: python sesi8/insight_bisnis.py
# Memakai data dan batch inference yang sama dengan app_final_dashboard.py.
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from utils.model_utils import load_model, load_metadata, predict

META, MODEL = load_metadata(), load_model()
raw = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
raw["total_charges"] = raw["total_charges"].fillna(raw["monthly_charges"] * raw["tenure_months"])
DF = predict(MODEL, raw, META)
T = DF[DF.risk_category == "Tinggi"]


def rp(x):
    return f"Rp{x:,.0f}".replace(",", ".")


def pct(x):
    return f"{x:.1%}".replace(".", ",")


def angka(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


porsi_tf = T.payment_method.value_counts() / DF.payment_method.value_counts()
porsi_reg = (DF.risk_category == "Tinggi").groupby(DF.region).mean().sort_values(ascending=False)
churn_per_risk = DF.groupby("risk_category")["churn"].mean()

insight = [
    (f"{len(T)} pelanggan ({pct(len(T) / len(DF))}) berada pada kategori risiko tinggi, "
     f"dengan total tagihan {rp(T.monthly_charges.sum())} per bulan.",
     "Jadikan kelompok ini fokus program retensi bulan ini."),
    (f"{pct((T.contract_type == 'Bulanan').mean())} pelanggan risiko tinggi berkontrak Bulanan.",
     "Program migrasi ke kontrak Tahunan dengan insentif diskon untuk pelanggan Bulanan berisiko tinggi."),
    (f"Rata-rata tenure pelanggan risiko tinggi {angka(T.tenure_months.mean())} bulan, dibanding "
     f"{angka(DF.tenure_months.mean())} bulan untuk seluruh pelanggan.",
     "Perkuat onboarding dan kontak proaktif pada 12 bulan pertama."),
    (f"Rata-rata keluhan pelanggan risiko tinggi {angka(T.num_complaints.mean(), 2)}, dibanding "
     f"{angka(DF.num_complaints.mean(), 2)} untuk seluruh pelanggan.",
     "SLA penyelesaian keluhan maksimal 48 jam untuk pelanggan berisiko tinggi."),
    ("Proporsi risiko tinggi per metode pembayaran: "
     + ", ".join(f"{k} {pct(v)}" for k, v in porsi_tf.sort_values(ascending=False).items()) + ".",
     "Dorong autodebit atau e-wallet untuk mengurangi friksi pembayaran."),
    (f"{porsi_reg.index[0]} memiliki proporsi risiko tinggi terbesar ({pct(porsi_reg.iloc[0])}), "
     f"sedangkan {porsi_reg.index[-1]} terendah ({pct(porsi_reg.iloc[-1])}).",
     f"Prioritaskan kampanye retensi regional di {porsi_reg.index[0]}."),
    (f"Kategori risiko konsisten dengan data historis: churn aktual {pct(churn_per_risk['Tinggi'])} pada "
     f"Tinggi, {pct(churn_per_risk['Sedang'])} pada Sedang, dan {pct(churn_per_risk['Rendah'])} pada Rendah.",
     "Sebagian data ini juga dipakai saat model dilatih; acuan kinerja yang adil adalah metrik holdout "
     f"(ROC-AUC {angka(META['holdout_metrics']['roc_auc'], 3)})."),
    (f"{(T.churn == 0).sum()} pelanggan yang masih aktif (belum churn) masuk kategori risiko tinggi.",
     f"Jadikan {(T.churn == 0).sum()} pelanggan ini daftar kerja pertama tim retensi minggu ini."),
]

if __name__ == "__main__":
    print("INSIGHT & REKOMENDASI — Customer Churn Intelligence Dashboard\n")
    for i, (temuan, saran) in enumerate(insight, 1):
        print(f"{i}. {temuan}\n   Rekomendasi: {saran}\n")
    out = ROOT / "sesi8" / "daftar_kerja_retensi.csv"
    cols = ["customer_id", "region", "segment", "contract_type", "payment_method", "tenure_months",
            "monthly_charges", "num_complaints", "churn_probability", "risk_category"]
    (T[T.churn == 0].sort_values("churn_probability", ascending=False)[cols]
        .to_csv(out, index=False))
    print(f"Daftar kerja retensi ({(T.churn == 0).sum()} pelanggan aktif berisiko tinggi) tersimpan di", out)
