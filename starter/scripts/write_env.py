"""Write starter/.env from the local pi auth store. Never committed (.gitignore).

    python starter/scripts/write_env.py
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUTH = Path.home() / ".pi" / "agent" / "auth.json"

key = json.loads(AUTH.read_text(encoding="utf-8"))["mistral"]["key"]
env = REPO_ROOT / "starter" / ".env"
env.write_text(
    f"MISTRAL_API_KEY={key}\nEMBEDDING_MODEL=mistral-embed\n",
    encoding="utf-8",
)
print(f"wrote {env} (key length {len(key)})")
