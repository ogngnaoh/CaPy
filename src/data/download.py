"""Download Rosetta CDRP-bio profiles from the Cell Painting Gallery.

WHY THIS WORKS
--------------
The Cell Painting Gallery on S3 is publicly accessible with
``--no-sign-request``, so no AWS credentials are required.  We try S3
first (faster, resume-capable) and fall back to HTTP.  Downloads are
idempotent: if a file already exists and meets a minimum size check, we
skip it.  This prevents re-downloading ~200 MB of profiles every run.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from urllib.request import urlretrieve

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# S3 / HTTP sources
# ---------------------------------------------------------------------------

S3_BASE = (
    "s3://cellpainting-gallery/cpg0003-rosetta/" "broad/workspace/preprocessed_data/"
)

# Morphology profiles (CellProfiler, replicate-level)
S3_MORPH = f"{S3_BASE}CDRPBIO-BBBC036-Bray/CellPainting/"
# Expression profiles (L1000, replicate-level)
S3_EXPR = f"{S3_BASE}CDRPBIO-BBBC036-Bray/L1000/"

# HTTP fallback — direct S3 HTTPS URLs
HTTP_MORPH = (
    "https://cellpainting-gallery.s3.amazonaws.com/"
    "cpg0003-rosetta/broad/workspace/preprocessed_data/"
    "CDRPBIO-BBBC036-Bray/CellPainting/"
    "replicate_level_cp_normalized_variable_selected.csv.gz"
)
HTTP_EXPR = (
    "https://cellpainting-gallery.s3.amazonaws.com/"
    "cpg0003-rosetta/broad/workspace/preprocessed_data/"
    "CDRPBIO-BBBC036-Bray/L1000/"
    "replicate_level_l1k.csv.gz"
)

# Broad CLUE compound metadata
HTTP_COMPOUND_META = (
    "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/"
    "repurposing_samples_20200324.txt"
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _file_exists_and_valid(path: Path, min_size_bytes: int = 1000) -> bool:
    """Check whether *path* exists and exceeds *min_size_bytes*.

    Args:
        path: File path to check.
        min_size_bytes: Minimum acceptable file size in bytes.

    Returns:
        ``True`` if the file exists and is large enough.
    """
    if not path.is_file():
        return False
    return path.stat().st_size >= min_size_bytes


def _download_via_s3(s3_path: str, local_path: Path) -> bool:
    """Download a file from S3 using the AWS CLI (no credentials needed).

    Args:
        s3_path: Full ``s3://`` URI.
        local_path: Destination on disk.

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "cp", "--no-sign-request", s3_path, str(local_path)]
    logger.info("S3 download: %s -> %s", s3_path, local_path)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("S3 download failed (%s), will try HTTP fallback.", exc)
        return False


def _download_via_http(url: str, local_path: Path) -> bool:
    """Download a file over HTTPS with a progress hook.

    Args:
        url: Source URL.
        local_path: Destination on disk.

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("HTTP download: %s -> %s", url, local_path)
    try:
        # Optional tqdm progress bar
        try:
            from tqdm import tqdm

            pbar = tqdm(unit="B", unit_scale=True, desc=local_path.name)

            def _reporthook(block_num: int, block_size: int, total: int) -> None:
                if pbar.total is None and total > 0:
                    pbar.total = total
                pbar.update(block_size)

            urlretrieve(url, str(local_path), reporthook=_reporthook)
            pbar.close()
        except ImportError:
            urlretrieve(url, str(local_path))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("HTTP download failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_rosetta_profiles(
    output_dir: str | Path,
    dry_run: bool = False,
    use_s3: bool = True,
) -> dict[str, Path]:
    """Download Rosetta CDRP-bio morphology and expression profiles.

    Args:
        output_dir: Root directory for raw data (e.g. ``data/raw``).
        dry_run: If ``True``, log what *would* be downloaded without
            fetching anything.
        use_s3: Try the AWS S3 CLI first; fall back to HTTP on failure.

    Returns:
        Dict mapping ``"morphology"`` and ``"expression"`` to their
        local file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "morphology": {
            "local": output_dir
            / "morphology"
            / "replicate_level_cp_normalized_variable_selected.csv.gz",
            "s3": f"{S3_MORPH}replicate_level_cp_normalized_variable_selected.csv.gz",
            "http": HTTP_MORPH,
        },
        "expression": {
            "local": output_dir / "expression" / "replicate_level_l1k.csv.gz",
            "s3": f"{S3_EXPR}replicate_level_l1k.csv.gz",
            "http": HTTP_EXPR,
        },
    }

    result: dict[str, Path] = {}
    for name, info in files.items():
        local = info["local"]
        if _file_exists_and_valid(local):
            logger.info("%s already downloaded: %s", name, local)
            result[name] = local
            continue

        if dry_run:
            logger.info("[DRY RUN] would download %s -> %s", name, local)
            result[name] = local
            continue

        success = False
        if use_s3:
            success = _download_via_s3(info["s3"], local)
        if not success:
            success = _download_via_http(info["http"], local)
        if not success:
            logger.error("Failed to download %s profile.", name)
        else:
            result[name] = local

    return result


def download_compound_metadata(
    output_dir: str | Path,
    dry_run: bool = False,
) -> Path:
    """Download compound metadata from the Broad Repurposing Hub.

    Args:
        output_dir: Root directory for raw data.
        dry_run: If ``True``, only log the planned download.

    Returns:
        Path to the downloaded metadata file.
    """
    output_dir = Path(output_dir)
    local = output_dir / "metadata" / "repurposing_samples.txt"

    if _file_exists_and_valid(local):
        logger.info("Compound metadata already downloaded: %s", local)
        return local

    if dry_run:
        logger.info("[DRY RUN] would download compound metadata -> %s", local)
        return local

    if _download_via_http(HTTP_COMPOUND_META, local):
        logger.info("Compound metadata saved to %s", local)
    else:
        logger.error("Failed to download compound metadata.")

    return local


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Rosetta CDRP-bio profiles.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Root output directory (default: data/raw)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log download plan without fetching files.",
    )
    parser.add_argument(
        "--no-s3",
        action="store_true",
        help="Skip S3 and download via HTTP only.",
    )
    args = parser.parse_args()
    download_rosetta_profiles(
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        use_s3=not args.no_s3,
    )
    download_compound_metadata(output_dir=args.output_dir, dry_run=args.dry_run)
