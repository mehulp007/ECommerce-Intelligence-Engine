"""Fetch and identify the immutable UCI Online Retail source workbook."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

SOURCE_URL = "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip"
ARCHIVE_NAME = "online_retail.zip"
WORKBOOK_NAME = "Online Retail.xlsx"
EXPECTED_SOURCE_ROWS = 541_909
EXPECTED_ARCHIVE_SHA256 = (
    "f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b"
)
EXPECTED_WORKBOOK_SHA256 = (
    "43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_workbook(
    raw_dir: Path, *, force_download: bool = False
) -> tuple[Path, dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / ARCHIVE_NAME
    workbook = raw_dir / WORKBOOK_NAME

    if force_download or not archive.is_file():
        temporary = raw_dir / f"{ARCHIVE_NAME}.part"
        try:
            request = urllib.request.Request(
                SOURCE_URL, headers={"User-Agent": "ecommerce-intelligence-engine/0.1"}
            )
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as target,
            ):
                shutil.copyfileobj(response, target)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)

    archive_sha256 = sha256_file(archive)
    if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("The UCI archive differs from the validated source checksum")

    with zipfile.ZipFile(archive) as source_zip:
        if source_zip.testzip() is not None:
            raise ValueError("The UCI download failed its ZIP integrity check")
        members = [item for item in source_zip.infolist() if not item.is_dir()]
        if len(members) != 1 or members[0].filename != WORKBOOK_NAME:
            raise ValueError("The UCI archive does not contain the expected workbook")
        temporary = raw_dir / f"{WORKBOOK_NAME}.part"
        try:
            with source_zip.open(members[0]) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(workbook)
        finally:
            temporary.unlink(missing_ok=True)

    workbook_sha256 = sha256_file(workbook)
    if workbook_sha256 != EXPECTED_WORKBOOK_SHA256:
        raise ValueError("The UCI workbook differs from the validated source checksum")

    return workbook, {
        "source_url": SOURCE_URL,
        "archive_sha256": archive_sha256,
        "workbook_sha256": workbook_sha256,
        "archive_bytes": archive.stat().st_size,
        "workbook_bytes": workbook.stat().st_size,
    }
