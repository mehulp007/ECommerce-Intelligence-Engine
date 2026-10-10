"""Locate, download, and unpack the H&M Personalized Fashion competition archive."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from ecommerce_intelligence.paths import ProjectPaths

COMPETITION = "h-and-m-personalized-fashion-recommendations"
CSV_MEMBERS = ("articles.csv", "customers.csv", "transactions_train.csv")
IMAGE_MEMBER = re.compile(r"^images/(?P<folder>\d{3})/(?P<article_id>\d{10})\.jpg$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(paths: ProjectPaths, *, force: bool = False) -> Path:
    """Download the archive with the Kaggle CLI using the user's own token."""
    paths.raw.mkdir(parents=True, exist_ok=True)
    if paths.archive.is_file() and not force:
        return paths.archive
    command = [
        sys.executable,
        "-m",
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(paths.raw),
    ]
    if force:
        command.append("--force")
    subprocess.run(command, check=True)
    if not paths.archive.is_file():
        raise FileNotFoundError(f"Download finished without {paths.archive.name}")
    return paths.archive


def extract_csvs(paths: ProjectPaths) -> dict[str, int]:
    """Copy the three CSV tables out of the archive; images stay zipped."""
    if not paths.archive.is_file():
        raise FileNotFoundError(
            "H&M archive not found. Run `eie data download` after joining the "
            "competition and saving a Kaggle API token."
        )
    paths.csv.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    with zipfile.ZipFile(paths.archive) as archive:
        members = {info.filename: info for info in archive.infolist()}
        missing = [name for name in CSV_MEMBERS if name not in members]
        if missing:
            raise ValueError(f"Archive is missing expected tables: {missing}")
        for name in CSV_MEMBERS:
            info = members[name]
            target = paths.csv / name
            if target.is_file() and target.stat().st_size == info.file_size:
                sizes[name] = info.file_size
                continue
            partial = target.with_suffix(".csv.part")
            try:
                # ZipFile verifies the member CRC once the stream is fully read.
                with archive.open(info) as source, partial.open("wb") as sink:
                    shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
            sizes[name] = info.file_size
    return sizes


def image_members(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Return (article_id, member name) pairs for well-formed image entries."""
    pairs = []
    for name in archive.namelist():
        match = IMAGE_MEMBER.match(name)
        if match and match["folder"] == match["article_id"][:3]:
            pairs.append((match["article_id"], name))
    return sorted(pairs)
