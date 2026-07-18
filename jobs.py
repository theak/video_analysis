"""Job store + background worker for the video-analysis web app.

Each job is a directory data/jobs/<job_id>/ holding metadata.json, log.txt,
the downloaded video, and every pipeline output (the job dir is passed to
analyze_video.py as --out, so the pipeline's per-stage caches double as
resume state after a crash or restart).
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO / "data"))
JOBS_DIR = DATA_DIR / "jobs"
PIPELINE = REPO / "scripts" / "analyze_video.py"

# high = the pipeline's own defaults (scene-threshold 0.05, interval 75s, no cap)
PRESETS = {
    "high": [],
    "medium": ["--scene-threshold", "0.15", "--interval", "150", "--max-frames", "30"],
    "low": ["--scene-threshold", "0.3", "--interval", "300", "--max-frames", "12"],
}

# Download resolution cap per detail level (yt-dlp -S res:N).
RESOLUTION = {"high": 1080, "medium": 720, "low": 480}

STAGES = {
    1: "extracting audio",
    2: "transcribing",
    3: "selecting keyframes",
    4: "extracting keyframes",
    5: "analyzing frames",
    6: "building report",
}
STAGE_RE = re.compile(r"^\[(\d)/6\]")
FRAME_RE = re.compile(r"^\s+\[(\d+)/(\d+)\]")

VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v", ".ts"}

EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("MAX_WORKERS", "1")))


def job_id_for(url: str, detail: str) -> str:
    return hashlib.sha256(f"{url}|{detail}".encode()).hexdigest()[:16]


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def load_meta(jdir: Path) -> dict:
    try:
        return json.loads((jdir / "metadata.json").read_text())
    except (OSError, ValueError):
        return {}


def update(jdir: Path, **fields) -> dict:
    meta = load_meta(jdir)
    meta.update(fields, updated_at=time.time())
    tmp = jdir / "metadata.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, jdir / "metadata.json")
    return meta


def list_jobs() -> list[dict]:
    jobs = [load_meta(p.parent) for p in JOBS_DIR.glob("*/metadata.json")]
    jobs = [m for m in jobs if m.get("id")]
    jobs.sort(key=lambda m: m.get("created_at", 0), reverse=True)
    return jobs


def find_video(jdir: Path) -> Path | None:
    for p in sorted(jdir.glob("video.*")):
        if p.suffix.lower() in VIDEO_SUFFIXES:
            return p
    return None


def find_thumbnail(jdir: Path) -> Path | None:
    thumb = jdir / "thumb.jpg"
    if thumb.exists():
        return thumb
    frames = sorted(jdir.glob("keyframes/*.jpg"))
    return frames[0] if frames else None


def submit(url: str, detail: str) -> dict:
    """Create (or reuse) a job for url+detail and enqueue it. Returns metadata."""
    jid = job_id_for(url, detail)
    jdir = job_dir(jid)
    meta = load_meta(jdir)
    if meta and meta.get("status") not in (None, "failed"):
        return meta  # already queued/running/done — dedupe
    jdir.mkdir(parents=True, exist_ok=True)
    meta = update(
        jdir,
        id=jid,
        url=url,
        detail=detail,
        status="queued",
        stage=None,
        error=None,
        title=meta.get("title"),
        created_at=meta.get("created_at") or time.time(),
    )
    EXECUTOR.submit(run_job, jid)
    return meta


def submit_upload(file_storage, detail: str) -> dict:
    """Save an uploaded video file as a job source and enqueue it. The absence
    of a `url` in metadata is what marks this a local upload (see run_job)."""
    suffix = Path(file_storage.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise ValueError(
            f"unsupported file type '{suffix or '?'}' — upload a video file "
            f"({', '.join(sorted(VIDEO_SUFFIXES))})"
        )
    jid = hashlib.sha256(
        f"{file_storage.filename}|{detail}|{time.time()}".encode()
    ).hexdigest()[:16]
    jdir = job_dir(jid)
    jdir.mkdir(parents=True, exist_ok=True)
    file_storage.save(jdir / f"video{suffix}")
    meta = update(
        jdir,
        id=jid,
        detail=detail,
        status="queued",
        stage=None,
        error=None,
        title=file_storage.filename,
        created_at=time.time(),
    )
    EXECUTOR.submit(run_job, jid)
    return meta


def delete_job(job_id: str) -> None:
    """Remove a job's directory (video, frames, transcript, report, metadata)."""
    jdir = job_dir(job_id)
    if jdir.exists():
        shutil.rmtree(jdir)


def submit_summaries(job_id: str) -> dict:
    """Enqueue deferred AI-summary generation for an already-analyzed job."""
    jdir = job_dir(job_id)
    meta = load_meta(jdir)
    if meta.get("summary_status") == "generating":
        return meta  # already in flight — dedupe
    meta = update(jdir, summaries="requested", summary_status="generating")
    EXECUTOR.submit(run_summaries, job_id)
    return meta


def recover_and_start() -> None:
    """Re-enqueue jobs interrupted by a restart; stage caches make this cheap."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    for meta in list_jobs():
        if meta.get("status") not in ("done", "failed"):
            EXECUTOR.submit(run_job, meta["id"])
        elif meta.get("summary_status") == "generating":
            EXECUTOR.submit(run_summaries, meta["id"])


def _run_logged(cmd: list[str], jdir: Path, on_line=None) -> int:
    """Run cmd, tee stdout+stderr to log.txt, call on_line per line."""
    with open(jdir / "log.txt", "a") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if on_line:
                on_line(line)
        return proc.wait()


def _pipeline_cmd(jdir: Path, meta: dict, video: Path, title: str,
                  skip_analysis: bool) -> list[str]:
    cmd = [sys.executable, "-u", str(PIPELINE), str(video),
           "--out", str(jdir), "--topic", title, "--title", title,
           *PRESETS[meta["detail"]]]
    if skip_analysis:
        cmd.append("--skip-analysis")
    return cmd


def _stage_parser(jdir: Path):
    def on_line(line: str):
        m = STAGE_RE.match(line)
        if m:
            update(jdir, stage=STAGES.get(int(m.group(1)), "processing"))
            return
        f = FRAME_RE.match(line)
        if f:
            update(jdir, stage=f"analyzing frames ({f.group(1)}/{f.group(2)})")
    return on_line


def run_summaries(jid: str) -> None:
    """Deferred stage 5: re-run the pipeline without --skip-analysis. Stages
    1-4 hit their caches, so only the LLM frame analysis actually runs."""
    jdir = job_dir(jid)
    meta = load_meta(jdir)
    try:
        video = find_video(jdir)
        if not video:
            raise RuntimeError("video file missing; cannot generate summaries")
        title = meta.get("title") or meta.get("url") or ""
        rc = _run_logged(_pipeline_cmd(jdir, meta, video, title, False),
                         jdir, _stage_parser(jdir))
        if rc != 0:
            raise RuntimeError("summary generation failed (see log)")
        update(jdir, summary_status="done", stage=None)
    except Exception as e:  # noqa: BLE001 — any failure marks summaries failed
        update(jdir, summary_status="failed", stage=None, error=str(e)[:500])


def run_job(jid: str) -> None:
    jdir = job_dir(jid)
    meta = load_meta(jdir)
    try:
        if meta.get("url"):
            # 1. Fetch metadata (title) once.
            info_path = jdir / "info.json"
            if not info_path.exists():
                update(jdir, status="downloading", stage="fetching metadata")
                out = subprocess.run(
                    ["yt-dlp", "-J", "--no-playlist", meta["url"]],
                    capture_output=True, text=True,
                )
                if out.returncode != 0:
                    with open(jdir / "log.txt", "a") as log:
                        log.write(out.stderr)
                    raise RuntimeError(_last_error_line(out.stderr) or "could not fetch video info")
                info = json.loads(out.stdout)
                info_path.write_text(json.dumps(
                    {"title": info.get("title"), "duration": info.get("duration"),
                     "uploader": info.get("uploader")}))
            info = json.loads(info_path.read_text())
            title = info.get("title") or meta["url"]
            update(jdir, title=title)

            # 2. Download video + thumbnail (yt-dlp resumes .part files itself).
            if not find_video(jdir):
                update(jdir, status="downloading", stage="downloading video")
                rc = _run_logged(
                    ["yt-dlp", "--no-playlist", "--newline",
                     "-f", "bv*+ba/b",
                     "-S", f"res:{RESOLUTION.get(meta['detail'], 1080)}",
                     "--merge-output-format", "mp4",
                     "--write-thumbnail", "--convert-thumbnails", "jpg",
                     "-o", "video.%(ext)s", "-o", "thumbnail:thumb.%(ext)s",
                     "--paths", str(jdir), meta["url"]],
                    jdir,
                )
                if rc != 0 or not find_video(jdir):
                    raise RuntimeError("video download failed (see log)")
            video = find_video(jdir)
        else:
            # Uploaded file — already on disk, no yt-dlp stages.
            video = find_video(jdir)
            if not video:
                raise RuntimeError("uploaded video file missing")
            title = meta.get("title") or video.name

        # 3. Run the analysis pipeline; parse its "[N/6]" stage markers.
        # AI summaries are deferred by default — generated on demand from the
        # report page — unless this job already requested them.
        update(jdir, status="processing", stage="starting analysis")
        skip = meta.get("summaries") != "requested"
        cmd = _pipeline_cmd(jdir, meta, video, title, skip)
        rc = _run_logged(cmd, jdir, _stage_parser(jdir))
        if rc != 0 or not (jdir / "report.json").exists():
            raise RuntimeError("analysis pipeline failed (see log)")

        update(jdir, status="done", stage=None, error=None)
    except Exception as e:  # noqa: BLE001 — any failure marks the job failed
        update(jdir, status="failed", stage=None, error=str(e)[:500])


def _last_error_line(stderr: str) -> str:
    lines = [l.strip() for l in (stderr or "").splitlines() if l.strip()]
    for line in reversed(lines):
        if line.startswith("ERROR"):
            return line[:200]
    return lines[-1][:200] if lines else ""
