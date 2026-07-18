import bisect
import json
import re

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
)

import jobs

app = Flask(__name__)
# Pick up template edits without a restart (one stat() per render) — report UX
# is meant to be iterated on and re-rendered over existing analyses.
app.config["TEMPLATES_AUTO_RELOAD"] = True
jobs.recover_and_start()

ID_RE = re.compile(r"^[0-9a-f]{16}$")
FRAME_NAME_RE = re.compile(r"^[\w.-]+\.jpg$")


def _fmt_mmss(t):
    t = int(round(t))
    return f"{t // 60:02d}:{t % 60:02d}"


SENTENCE_END_RE = re.compile(r'[.!?]["\')\]]*$')


def _partition_transcript(jdir, frames):
    """Give each displayed frame the speech from its timestamp up to the next
    frame's, so the transcript view reads through the video exactly once.
    (The stored per-frame snippets use overlapping ±windows — right for LLM
    context, but repetitive when read as the primary transcript view.)

    Splits on sentence boundaries, not raw segment times: a sentence belongs
    to the frame it started in, even if it ends after the next frame's
    timestamp. Word times are interpolated within each Whisper segment."""
    tpath = jdir / "transcript.json"
    if not frames or not tpath.exists():
        return
    try:
        segments = json.loads(tpath.read_text()).get("segments") or []
    except (OSError, ValueError):
        return

    words = []  # (approx start time, word)
    for s in segments:
        toks = s["text"].split()
        if not toks:
            continue
        step = (s["end"] - s["start"]) / len(toks)
        words.extend((s["start"] + i * step, w) for i, w in enumerate(toks))

    sentences = []  # (start time, text)
    cur, start = [], None
    for t, w in words:
        if start is None:
            start = t
        cur.append(w)
        if SENTENCE_END_RE.search(w):
            sentences.append((start, " ".join(cur)))
            cur, start = [], None
    if cur:
        sentences.append((start, " ".join(cur)))

    times = [f["t"] for f in frames]
    parts = [[] for _ in frames]
    for st, text in sentences:
        parts[max(bisect.bisect_right(times, st) - 1, 0)].append(text)
    for f, texts in zip(frames, parts):
        f["transcript"] = " ".join(texts).strip()


def _job_dir_or_404(jid):
    if not ID_RE.fullmatch(jid):
        abort(404)
    jdir = jobs.job_dir(jid)
    if not (jdir / "metadata.json").exists():
        abort(404)
    return jdir


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    if file and file.filename:
        detail = (request.form.get("detail") or "medium").strip().lower()
        if detail not in jobs.PRESETS:
            return jsonify({"error": "detail must be low, medium, or high"}), 400
        try:
            meta = jobs.submit_upload(file, detail)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"id": meta["id"], "status": meta["status"]})

    data = request.get_json(silent=True) or request.form
    url = (data.get("url") or "").strip()
    detail = (data.get("detail") or "medium").strip().lower()
    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "Please enter a valid http(s) video URL."}), 400
    if detail not in jobs.PRESETS:
        return jsonify({"error": "detail must be low, medium, or high"}), 400
    meta = jobs.submit(url, detail)
    return jsonify({"id": meta["id"], "status": meta["status"]})


@app.route("/api/videos")
def api_videos():
    out = []
    for m in jobs.list_jobs():
        out.append({
            "id": m["id"],
            "url": m.get("url"),
            "title": m.get("title"),
            "detail": m.get("detail"),
            "status": m.get("status"),
            "stage": m.get("stage"),
            "error": m.get("error"),
            "created_at": m.get("created_at"),
        })
    return jsonify(out)


@app.route("/api/videos/<jid>", methods=["DELETE"])
def delete_video(jid):
    _job_dir_or_404(jid)
    jobs.delete_job(jid)
    return jsonify({"ok": True})


@app.route("/video/<jid>")
def video(jid):
    """Render report.json server-side — presentation lives in the template, so
    UX changes apply retroactively to every already-analyzed video."""
    jdir = _job_dir_or_404(jid)
    rpath = jdir / "report.json"
    if not rpath.exists():
        if (jdir / "report.html").exists():  # legacy self-contained report
            return send_from_directory(jdir, "report.html")
        return redirect("/")
    data = json.loads(rpath.read_text())
    report_frames = data.get("frames", [])
    has_summaries = any(f.get("analysis") for f in report_frames)
    frames = []
    for f in report_frames:
        analysis = f.get("analysis")
        # relevance filtering only applies once a frame has been analyzed;
        # transcript-only frames are always shown
        if analysis and not analysis.get("relevant", True):
            continue
        analysis = analysis or {}
        frames.append({
            "t": f.get("t", 0),
            "t_label": _fmt_mmss(f.get("t", 0)),
            "image_name": (f.get("image") or "").rsplit("/", 1)[-1],
            "title": str(analysis.get("title") or ""),
            "bullets": [str(b) for b in analysis.get("bullets") or []],
            "snippet": f.get("snippet") or "",
            "transcript": f.get("snippet") or "",  # fallback if no transcript.json
        })
    _partition_transcript(jdir, frames)
    meta = jobs.load_meta(jdir)
    return render_template(
        "report.html",
        job_id=jid,
        title=data.get("title") or meta.get("title") or "Video analysis",
        model=data.get("model") or "",
        frames=frames,
        has_summaries=has_summaries,
        generating=(not has_summaries
                    and meta.get("summary_status") == "generating"),
    )


@app.route("/frame/<jid>/<name>")
def frame(jid, name):
    jdir = _job_dir_or_404(jid)
    if not FRAME_NAME_RE.fullmatch(name):
        abort(404)
    resp = send_from_directory(jdir / "keyframes", name)
    resp.headers["Cache-Control"] = "public, max-age=86400"  # frames never change
    return resp


def _summary_status(jdir):
    """done = report.json actually has analyses; metadata covers the rest."""
    rpath = jdir / "report.json"
    if rpath.exists():
        data = json.loads(rpath.read_text())
        if any(f.get("analysis") for f in data.get("frames", [])):
            return "done"
    return jobs.load_meta(jdir).get("summary_status") or "none"


@app.route("/api/videos/<jid>/summarize", methods=["POST"])
def summarize(jid):
    jdir = _job_dir_or_404(jid)
    status = _summary_status(jdir)
    if status in ("done", "generating"):
        return jsonify({"status": status})
    jobs.submit_summaries(jid)
    return jsonify({"status": "generating"})


@app.route("/api/videos/<jid>/summary_status")
def summary_status(jid):
    jdir = _job_dir_or_404(jid)
    return jsonify({"status": _summary_status(jdir)})


@app.route("/api/report/<jid>")
def api_report(jid):
    jdir = _job_dir_or_404(jid)
    if not (jdir / "report.json").exists():
        abort(404)
    return send_from_directory(jdir, "report.json")


@app.route("/thumb/<jid>")
def thumb(jid):
    jdir = _job_dir_or_404(jid)
    path = jobs.find_thumbnail(jdir)
    if not path:
        return redirect("/static/placeholder.svg")
    resp = send_from_directory(jdir, path.relative_to(jdir).as_posix())
    resp.headers["Cache-Control"] = "public, max-age=60"
    return resp


@app.route("/log/<jid>")
def log(jid):
    jdir = _job_dir_or_404(jid)
    if not (jdir / "log.txt").exists():
        abort(404)
    resp = send_from_directory(jdir, "log.txt")
    resp.mimetype = "text/plain"
    return resp


if __name__ == "__main__":
    app.run(debug=True, port=5177, use_reloader=False)
