from pathlib import Path
from datetime import datetime
import uuid

TMP_DIR = Path("data/tmp_uploads")

def ensure_tmp_dir():
    TMP_DIR.mkdir(parents=True, exist_ok=True)

def make_temp_path(original_filename: str) -> Path:
    ensure_tmp_dir()
    suffix = Path(original_filename).suffix.lower()
    name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}{suffix}"
    return TMP_DIR / name
