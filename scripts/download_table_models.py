"""下载离线 Table Transformer 权重到 .models/。"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download  # noqa: E402


def main() -> int:
    for repo, local in (
        ("microsoft/table-transformer-detection", ".models/table-transformer-detection"),
        (
            "microsoft/table-transformer-structure-recognition-v1.1-all",
            ".models/table-transformer-structure",
        ),
    ):
        target = ROOT / local
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo,
            local_dir=str(target),
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt"],
        )
        print("ok", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
