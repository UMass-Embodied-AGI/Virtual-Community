"""Asset utilities for downloading and managing ViCo assets from HuggingFace.

Asset Search Order:
1. VICO_ASSET_PATHS environment variable (colon-separated paths)
2. Local cache at vico/assets/ViCo/ (downloaded on first use)
3. Download from HuggingFace

All ViCo assets are stored in Virtual-Community-AI/assets.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

VICO_ASSETS_REPO = "Virtual-Community-AI/assets"

_VICO_ASSETS_DIR = Path(__file__).parent.parent / "assets" / "ViCo"


def get_asset_path(
    asset_name: str,
    repo_id: str = VICO_ASSETS_REPO,
    pattern_is_dir: bool = True,
    revision: str | None = None,
) -> str:
    """Get local path to a ViCo asset, downloading from HuggingFace if needed.

    1. Check VICO_ASSET_PATHS environment variable for local override
    2. Download from HuggingFace (incremental — only fetches missing files)

    Args:
        asset_name: Relative path within the repository (e.g., "scene/v1/NY")
        repo_id: HuggingFace dataset repository ID
        pattern_is_dir: If True, download all files under asset_name/
        revision: Optional specific revision/commit to use

    Returns:
        Absolute path to the local asset file or directory
    """
    # 1. Check environment variable for local override
    local_override = os.environ.get("VICO_ASSET_PATHS", "")
    if local_override:
        for dir_path in local_override.split(":"):
            dir_path = dir_path.strip()
            if dir_path and os.path.exists(os.path.join(dir_path, asset_name)):
                return os.path.join(dir_path, asset_name)

    # 2. Download from HuggingFace (incremental caching via snapshot_download)
    local_path = _VICO_ASSETS_DIR / asset_name
    if local_path.exists():
        return str(local_path)

    allow_patterns = f"{asset_name}/*" if pattern_is_dir else asset_name
    snapshot_download(
        repo_type="dataset",
        repo_id=repo_id,
        allow_patterns=allow_patterns,
        local_dir=str(_VICO_ASSETS_DIR),
        revision=revision,
    )
    return str(local_path)


def download_all_assets() -> None:
    """Download all ViCo assets from HuggingFace in one shot."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_type="dataset",
        repo_id=VICO_ASSETS_REPO,
        local_dir=str(_VICO_ASSETS_DIR),
    )


def ensure_asset(asset_name: str) -> str:
    """Download a ViCo asset lazily and return its absolute local path.

    - URDFs: downloads the entire parent directory (they reference sibling files).
    - Everything else: downloads the single file; if the file is still missing
      afterward (e.g. a texture inside a UUID bundle), retries by downloading
      the parent directory.

    Args:
        asset_name: Relative path within the ViCo assets repo (no leading ViCo/)

    Returns:
        Absolute path to the local asset file
    """
    local_path = _VICO_ASSETS_DIR / asset_name
    if local_path.exists():
        # OBJ files reference sibling .mtl and texture files — download the
        # parent directory if those companions are missing.
        if asset_name.endswith(".obj") and "/" in asset_name:
            mtl_name = local_path.stem + ".mtl"
            if not (local_path.parent / mtl_name).exists():
                get_asset_path(asset_name.rsplit("/", 1)[0])
        return str(local_path)

    if asset_name.endswith(".urdf"):
        get_asset_path(asset_name.rsplit("/", 1)[0])
    else:
        get_asset_path(asset_name, pattern_is_dir=False)
        if not local_path.exists() and "/" in asset_name:
            get_asset_path(asset_name.rsplit("/", 1)[0])

    return str(local_path)
