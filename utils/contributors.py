"""utils/contributors.py — daftar kontributor repo dari riwayat git (untuk halaman Team).

Sumbernya `git log`, bukan API GitHub: API GitHub hanya menghitung commit yang email-nya terhubung
ke akun GitHub, sehingga sebagian anggota tim tidak muncul. Identitas ganda (nama/email berbeda untuk
orang yang sama) digabung lewat file .mailmap di root repo.
"""
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEP = "\x1f"
NOREPLY = re.compile(r"^(?:\d+\+)?([A-Za-z0-9-]+)@users\.noreply\.github\.com$")


def github_login(email: str) -> str | None:
    """Username GitHub dari email noreply (mis. 123+nama@users.noreply.github.com), jika ada."""
    m = NOREPLY.match(email or "")
    return m.group(1) if m else None


def load_contributors(repo=ROOT) -> pd.DataFrame:
    """Satu baris per kontributor, terurut dari commit terbanyak.

    Commit merge (mis. "Merge pull request") tidak dihitung karena bukan pekerjaan baru.
    Mengembalikan DataFrame kosong jika git atau folder .git tidak tersedia (mis. saat deploy).
    """
    cols = ["name", "github", "commits", "first_commit", "last_commit", "recent"]
    try:
        out = subprocess.run(
            ["git", "log", "--no-merges", "--use-mailmap", "--date=short",
             f"--format=%aN{SEP}%aE{SEP}%ad{SEP}%s"],
            cwd=repo, capture_output=True, text=True, encoding="utf-8", timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return pd.DataFrame(columns=cols)
    rows = [line.split(SEP, 3) for line in out.splitlines() if line.count(SEP) == 3]
    if not rows:
        return pd.DataFrame(columns=cols)
    log = pd.DataFrame(rows, columns=["name", "email", "date", "subject"])  # terbaru di atas
    log["github"] = log["email"].map(github_login)
    team = (log.groupby("name", sort=False)
               .agg(github=("github", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
                    commits=("subject", "size"), first_commit=("date", "min"),
                    last_commit=("date", "max"), recent=("subject", lambda s: list(s[:3])))
               .reset_index())
    team["github"] = team["github"].astype(object).where(team["github"].notna(), None)
    return team.sort_values(["commits", "first_commit"], ascending=[False, True],
                            ignore_index=True)[cols]
