# Vercel entry point: exposes the Dash app's Flask server as `app`.
# Vercel only looks for app.py/index.py/main.py/... in the repo root (or src/, app/), so the
# dashboard in sesi8/ is imported from here. Run locally with: python sesi8/app_final_dashboard.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "sesi8"))

from app_final_dashboard import server  # noqa: E402

app = server
