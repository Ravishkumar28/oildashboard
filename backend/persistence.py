"""Optional HuggingFace Datasets persistence for the paper-trading book.

HF Spaces free tier has no persistent volume: the container filesystem is
ephemeral, so every code redeploy wipes ``paper_state.json`` and resets the
equity curve to $100k. To survive redeploys we periodically push the same
JSON to a HF Dataset repo and pull it back on startup.

If ``HF_TOKEN`` or ``PAPER_STATE_REPO`` env vars are missing — or if any HF
API call fails — we silently fall back to local-only state."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

REMOTE_FILENAME = "paper_state.json"


def download_state(repo_id: str, token: str) -> Optional[dict]:
    """Return the last persisted paper state from the dataset, or None."""
    if not (HF_AVAILABLE and repo_id and token):
        return None
    try:
        path = hf_hub_download(repo_id=repo_id, filename=REMOTE_FILENAME,
                               repo_type="dataset", token=token,
                               cache_dir=tempfile.gettempdir())
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def upload_state(repo_id: str, token: str, data: dict) -> bool:
    """Push the current paper state to the dataset. Returns True on success."""
    if not (HF_AVAILABLE and repo_id and token):
        return False
    try:
        api = HfApi(token=token)
        # write to a temp file in the system temp dir
        fd, tmp_path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=REMOTE_FILENAME,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="auto: persist paper trading state",
            )
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return True
    except Exception:
        return False
