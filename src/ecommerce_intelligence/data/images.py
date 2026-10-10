"""Turn the archive's full-size product photos into small WebP thumbnails."""

from __future__ import annotations

import io
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ecommerce_intelligence.data.source import image_members
from ecommerce_intelligence.paths import ProjectPaths

THUMBNAIL_BOX = (256, 384)
WEBP_QUALITY = 80
CHUNK_SIZE = 400


@dataclass(frozen=True)
class ThumbnailJob:
    archive: Path
    output_dir: Path
    members: tuple[tuple[str, str], ...]


def thumbnail_path(output_dir: Path, article_id: str) -> Path:
    return output_dir / article_id[:3] / f"{article_id}.webp"


def make_thumbnail(data: bytes, target: Path) -> None:
    with Image.open(io.BytesIO(data)) as image:
        # JPEG draft mode decodes at a reduced scale, which is much faster.
        image.draft("RGB", (THUMBNAIL_BOX[0] * 2, THUMBNAIL_BOX[1] * 2))
        image = image.convert("RGB")
        image.thumbnail(THUMBNAIL_BOX, Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".webp.part")
        image.save(partial, format="WEBP", quality=WEBP_QUALITY, method=4)
        partial.replace(target)


def _run_job(job: ThumbnailJob) -> tuple[int, int, list[str]]:
    created = skipped = 0
    failed: list[str] = []
    with zipfile.ZipFile(job.archive) as archive:
        for article_id, member in job.members:
            target = thumbnail_path(job.output_dir, article_id)
            if target.is_file():
                skipped += 1
                continue
            try:
                make_thumbnail(archive.read(member), target)
                created += 1
            except Exception:  # One unreadable photo must not stop the run.
                failed.append(article_id)
    return created, skipped, failed


def build_thumbnails(
    paths: ProjectPaths, *, workers: int | None = None, limit: int | None = None
) -> dict:
    if not paths.archive.is_file():
        raise FileNotFoundError("H&M archive not found. Run `eie data download`.")
    with zipfile.ZipFile(paths.archive) as archive:
        members = image_members(archive)
    if limit is not None:
        members = members[:limit]
    jobs = [
        ThumbnailJob(
            paths.archive, paths.thumbnails, tuple(members[i : i + CHUNK_SIZE])
        )
        for i in range(0, len(members), CHUNK_SIZE)
    ]
    created = skipped = 0
    failed: list[str] = []
    if workers == 1:
        results = map(_run_job, jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = (
            future.result()
            for future in as_completed([executor.submit(_run_job, j) for j in jobs])
        )
    try:
        for job_created, job_skipped, job_failed in results:
            created += job_created
            skipped += job_skipped
            failed.extend(job_failed)
    finally:
        if workers != 1:
            executor.shutdown()

    files = list(paths.thumbnails.rglob("*.webp")) if paths.thumbnails.is_dir() else []
    summary = {
        "archive_images": len(members),
        "created": created,
        "skipped_existing": skipped,
        "failed": len(failed),
        "failed_article_ids": sorted(failed)[:50],
        "thumbnail_files": len(files),
        "thumbnail_bytes": sum(path.stat().st_size for path in files),
        "box": list(THUMBNAIL_BOX),
        "format": f"webp q{WEBP_QUALITY}",
    }
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "thumbnails.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
