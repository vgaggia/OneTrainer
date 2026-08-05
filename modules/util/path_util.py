import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def safe_filename(
        text: str,
        allow_spaces: bool = True,
        max_length: int | None = 32,
):
    legal_chars = [' ', '.', '_', '-', '#']
    if not allow_spaces:
        text = text.replace(' ', '_')

    text = ''.join(filter(lambda x: str.isalnum(x) or x in legal_chars, text)).strip()

    if max_length is not None:
        text = text[0: max_length]

    return text.strip()


def canonical_join(base_path: str, *paths: str):
    # Creates a canonical path name that can be used for comparisons.
    # Also, Windows does understand / instead of \, so these paths can be used as usual.

    joined = os.path.join(base_path, *paths)
    return joined.replace('\\', '/')


def write_json_atomic(path: str, obj: Any):
    destination = Path(path)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".write",
                delete=False,
        ) as file:
            temp_path = Path(file.name)
            json.dump(obj, file, indent=4)
            file.flush()
            os.fsync(file.fileno())

        # Windows denies os.replace while another process briefly holds the destination open
        # (antivirus/indexers are common culprits). Retry sharing violations without giving up the
        # atomic same-directory replacement.
        for attempt in range(6):
            try:
                os.replace(temp_path, destination)
                temp_path = None
                return
            except PermissionError:  # noqa: PERF203 - retry loop is intentional on Windows
                if attempt == 5:
                    raise
                time.sleep(0.05 * (2 ** attempt))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


SUPPORTED_IMAGE_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.avif'}
SUPPORTED_VIDEO_EXTENSIONS = {'.webm', '.mkv', '.flv', '.avi', '.mov', '.wmv', '.mp4', '.mpeg', '.m4v'}
SUPPORTED_CAPTION_EXTENSIONS = {'.txt'}


def supported_image_extensions() -> set[str]:
    return SUPPORTED_IMAGE_EXTENSIONS


def is_supported_image_extension(extension: str) -> bool:
    return extension.lower() in SUPPORTED_IMAGE_EXTENSIONS


def supported_video_extensions() -> set[str]:
    return SUPPORTED_VIDEO_EXTENSIONS


def is_supported_video_extension(extension: str) -> bool:
    return extension.lower() in SUPPORTED_VIDEO_EXTENSIONS


def supported_caption_extensions() -> set[str]:
    return SUPPORTED_CAPTION_EXTENSIONS


def json_path_modifier(x: str | Path) -> Path:
    x = Path(x).absolute()
    return x.parent if x.suffix == ".json" else x
