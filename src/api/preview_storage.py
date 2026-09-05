"""Persist live observation previews outside SQLite.

Preview encoders remain with their existing callers.  This module is the
boundary between those in-memory data URIs and the live detection record:
after a detection has an id, PNG data URIs are atomically written to the
gitignored live cache and the observation fields become API references.
"""

import base64
import binascii
import os
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np


PREVIEWS_RELATIVE = Path("data") / "raw" / "live" / "previews"
MAX_PREVIEW_FILES = 500

# Detection ids are integers and these are the only names this module creates.
_OWNED_FILENAME = re.compile(
    r"^(?P<id>[0-9]+)_(?:original_(?:sar|overlay)|temporal_[0-9]+_(?:sar|overlay)|optical_rgb)\.png$"
)
_DATA_URI = re.compile(r"^data:image/png;base64,(?P<data>[A-Za-z0-9+/=\s]+)$")


def preview_directory(base_dir):
    """Return the live preview directory for an application root."""
    return Path(base_dir) / PREVIEWS_RELATIVE


def _directory_is_safe(directory, base_dir):
    """Reject a cache whose existing ancestor is a symlink or escapes root."""
    directory = Path(directory)
    base = Path(base_dir).resolve(strict=False)
    resolved = directory.resolve(strict=False)
    try:
        if os.path.commonpath((str(base), str(resolved))) != str(base):
            return False
    except ValueError:
        return False
    current = directory
    while True:
        if current.is_symlink():
            return False
        if current == Path(base_dir) or current.parent == current:
            break
        current = current.parent
    return True


def is_safe_preview_directory(directory, base_dir):
    return _directory_is_safe(directory, base_dir)


def _decode_png_data_uri(value):
    if not isinstance(value, str):
        return None
    match = _DATA_URI.fullmatch(value)
    if not match:
        return None
    try:
        data = base64.b64decode(match.group("data"), validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("invalid PNG data URI")
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("invalid PNG signature")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ValueError("invalid PNG pixels")
    return data


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise OSError("preview directory is a symlink")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _owned_pngs(directory):
    try:
        if directory.is_symlink():
            return []
        entries = list(directory.iterdir())
    except (FileNotFoundError, OSError):
        return []
    owned = []
    for entry in entries:
        if not _OWNED_FILENAME.fullmatch(entry.name) or entry.is_symlink():
            continue
        try:
            if entry.is_file():
                owned.append((entry, entry.stat().st_mtime))
        except OSError:
            continue
    return owned


def cleanup_previews(directory=None):
    """Evict oldest owned regular PNGs until the configured file cap holds."""
    directory = Path(directory) if directory is not None else None
    if directory is None:
        return
    try:
        owned = sorted(_owned_pngs(directory), key=lambda item: (item[1], item[0].name))
    except (FileNotFoundError, OSError):
        return
    excess = max(0, len(owned) - MAX_PREVIEW_FILES)
    for path, _ in owned[:excess]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # Cleanup is best effort and must not turn a successful detection
            # into an API failure.
            continue


def _store_observation(observation, detection_id, kind, directory):
    """Store the known preview fields in one observation, best effort."""
    field_to_suffix = {
        "sar_preview": f"{kind}_sar",
        "overlay_preview": f"{kind}_overlay",
        "rgb_preview": "optical_rgb" if kind == "optical" else None,
    }
    for field, suffix in field_to_suffix.items():
        if suffix is None or field not in observation:
            continue
        try:
            data = _decode_png_data_uri(observation[field])
            if data is None:
                # URL references are already persisted.  Any data URI reaching
                # this boundary is a new inline preview and must be removed if
                # it is not a valid PNG (never fall back to DB base64).
                if isinstance(observation[field], str) and observation[field].startswith("data:"):
                    raise ValueError("unsupported preview data URI")
                continue
            filename = f"{detection_id}_{suffix}.png"
            _atomic_write(directory / filename, data)
            observation[field] = f"/api/previews/{filename}"
        except Exception as exc:
            # Never retain a large inline fallback after a disk failure.  The
            # UI can show this honest, type-only error beside the observation.
            observation.pop(field, None)
            observation["preview_error"] = f"Preview storage failed ({type(exc).__name__})."


def persist_previews(supplementary, detection_id, base_dir):
    """Store all known live previews and return the mutated payload.

    The function intentionally catches storage errors.  Preview persistence is
    supplementary: a disk-full or permission issue must not undo a successful
    primary inference and DB INSERT.
    """
    directory = preview_directory(base_dir)
    if not _directory_is_safe(directory, base_dir):
        discard_inline_previews(supplementary, "OSError")
        return supplementary
    try:
        # Original is always one observation; temporal may contain multiple.
        original = supplementary.get("original")
        if isinstance(original, dict):
            _store_observation(original, detection_id, "original", directory)
        temporal = supplementary.get("temporal", {})
        observations = temporal.get("observations", []) if isinstance(temporal, dict) else []
        for index, observation in enumerate(observations):
            if isinstance(observation, dict):
                _store_observation(observation, detection_id, f"temporal_{index}", directory)
        optical = supplementary.get("optical", {})
        optical_observation = optical.get("observation") if isinstance(optical, dict) else None
        if isinstance(optical_observation, dict):
            _store_observation(optical_observation, detection_id, "optical", directory)
    except Exception as exc:
        # Include an honest error in the payload if setup itself fails before
        # an individual observation can report one.
        supplementary.setdefault("preview_error", f"Preview storage failed ({type(exc).__name__}).")
    finally:
        try:
            cleanup_previews(directory)
        except Exception:
            # Eviction is best effort and must never change primary success.
            pass
    return supplementary


def discard_inline_previews(supplementary, error_type="OSError"):
    """Remove data-URI fallbacks after an unexpected storage-boundary error."""
    observations = []
    original = supplementary.get("original")
    if isinstance(original, dict):
        observations.append(original)
    temporal = supplementary.get("temporal", {})
    if isinstance(temporal, dict):
        observations.extend(item for item in temporal.get("observations", []) if isinstance(item, dict))
    optical = supplementary.get("optical", {})
    if isinstance(optical, dict) and isinstance(optical.get("observation"), dict):
        observations.append(optical["observation"])
    for observation in observations:
        for field in ("sar_preview", "overlay_preview", "rgb_preview"):
            if isinstance(observation.get(field), str) and observation[field].startswith("data:"):
                observation.pop(field, None)
                observation["preview_error"] = f"Preview storage failed ({error_type})."


def is_safe_preview_filename(filename):
    """Accept only a generated basename, never paths or symlinks."""
    return isinstance(filename, str) and _OWNED_FILENAME.fullmatch(filename) is not None
