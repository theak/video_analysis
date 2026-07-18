#!/usr/bin/env python3
"""
Turn any video into structured report data: keyframes + concise LLM takeaways.

Pipeline (each stage caches to the output dir, so reruns skip completed work):
  1. Extract audio                (ffmpeg)
  2. Transcribe                   (OpenAI whisper-1, verbose_json; auto-chunks if >24MB)
  3. Select keyframe timestamps   (ffmpeg scene detection + fixed-interval coverage floor)
  4. Extract one keyframe / ts    (ffmpeg; timestamp embedded in filename)
  5. Analyze each keyframe        (vision model; image + aligned transcript -> JSON bullets)
  6. Write report.json            (frames + snippets + analysis; the Flask web app renders it)

Requires: ffmpeg + ffprobe on PATH; an OpenAI key in OPENAI_KEY or OPENAI_API_KEY.

Run:
  uv run --with openai python analyze_video.py INPUT.mp4 --topic "Acme's product"
"""

import argparse
import base64
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

SKILL_DIR = Path(__file__).resolve().parent.parent


# --- OpenAI client --------------------------------------------------------
def make_client():
    key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("Set OPENAI_KEY or OPENAI_API_KEY in the environment.")
    return OpenAI(api_key=key)


# --- Shell helper ---------------------------------------------------------
def run(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}")
    return proc


def fmt_mmss(t):
    t = int(round(t))
    return f"{t // 60:02d}:{t % 60:02d}"


def video_duration(video):
    p = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)])
    return float(p.stdout.strip())


# --- Stage 1: extract audio ----------------------------------------------
def extract_audio(video, out, force):
    audio = out / "audio.mp3"
    if audio.exists() and not force:
        print(f"[1/6] audio cached -> {audio.name}")
        return audio
    print("[1/6] extracting audio ...")
    # mono / 16kHz / 32kbps -> ~100 min per 24MB; tiny and speech-friendly for Whisper.
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-b:a", "32k", str(audio)])
    print(f"      audio.mp3 = {audio.stat().st_size / 1e6:.1f} MB")
    return audio


# --- Stage 2: transcribe (auto-chunks past Whisper's 25MB limit) ---------
CHUNK_SECONDS = 5400  # ~90 min at 32kbps mono ≈ 21MB, safely under the 25MB API limit


def _transcribe_file(client, model, path):
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=model, file=f, response_format="verbose_json")
    data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    return data


def transcribe(client, audio, out, model, force):
    tpath = out / "transcript.json"
    if tpath.exists() and not force:
        print(f"[2/6] transcript cached -> {tpath.name}")
        return json.loads(tpath.read_text())

    dur = video_duration(audio)
    n_chunks = max(1, math.ceil(dur / CHUNK_SECONDS))
    print(f"[2/6] transcribing with {model} ({n_chunks} chunk(s)) ...")

    segments, full_text = [], []
    for i in range(n_chunks):
        offset = i * CHUNK_SECONDS
        if n_chunks == 1:
            chunk = audio
        else:
            chunk = out / f"_chunk_{i}.mp3"
            run(["ffmpeg", "-y", "-ss", str(offset), "-t", str(CHUNK_SECONDS),
                 "-i", str(audio), "-c", "copy", str(chunk)])
        data = _transcribe_file(client, model, chunk)
        full_text.append(data.get("text", ""))
        for s in (data.get("segments") or []):
            segments.append({"start": s["start"] + offset,
                             "end": s["end"] + offset,
                             "text": s["text"].strip()})
        if chunk != audio:
            chunk.unlink(missing_ok=True)

    result = {"text": " ".join(full_text).strip(), "segments": segments}
    tpath.write_text(json.dumps(result, indent=2))
    print(f"      {len(segments)} segments")
    return result


# --- Stage 3: select keyframe timestamps ---------------------------------
def select_timestamps(video, out, scene_threshold, interval_s, min_gap_s,
                      max_frames, force):
    cache = out / "keyframe_timestamps.json"
    if cache.exists() and not force:
        ts = json.loads(cache.read_text())
        print(f"[3/6] keyframe timestamps cached -> {len(ts)} frames")
        return ts

    print("[3/6] selecting keyframes (scene changes + interval floor) ...")
    proc = subprocess.run(
        ["ffmpeg", "-i", str(video),
         "-vf", f"select='gt(scene,{scene_threshold})',showinfo",
         "-an", "-f", "null", "-"],
        capture_output=True, text=True)
    scene = sorted(float(m) for m in re.findall(r"pts_time:([0-9.]+)", proc.stderr))

    dur = video_duration(video)
    interval = [float(t) for t in range(0, int(dur), interval_s)] if interval_s > 0 else []

    # Merge scene cuts + interval floor + opening frame; dedupe within min_gap_s.
    kept = []
    for t in sorted(set(scene) | set(interval) | {0.0}):
        if not kept or t - kept[-1] >= min_gap_s:
            kept.append(t)

    # Optional cost cap: evenly subsample down to max_frames.
    if max_frames and len(kept) > max_frames:
        step = len(kept) / max_frames
        kept = [kept[int(i * step)] for i in range(max_frames)]
        print(f"      capped to {len(kept)} frames (--max-frames)")

    cache.write_text(json.dumps(kept, indent=2))
    print(f"      {len(scene)} scene cuts + {len(interval)} interval samples "
          f"-> {len(kept)} kept (>= {min_gap_s}s apart)")
    return kept


# --- Stage 4: extract keyframes ------------------------------------------
def extract_keyframes(video, timestamps, out, force):
    kdir = out / "keyframes"
    kdir.mkdir(exist_ok=True)
    print("[4/6] extracting keyframes ...")
    frames = []
    for t in timestamps:
        secs = int(round(t))
        name = f"keyframe_{secs // 60:02d}{secs % 60:02d}_{t:.2f}s.jpg"
        path = kdir / name
        if not path.exists() or force:
            run(["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(video),
                 "-frames:v", "1", "-q:v", "3", str(path)])
        frames.append({"t": t, "path": path})
    print(f"      {len(frames)} keyframes -> {kdir.name}/")
    return frames


# --- Stage 5: transcript alignment + LLM analysis ------------------------
def snippet_for(t, segments, before, after):
    lo, hi = t - before, t + after
    return " ".join(s["text"] for s in segments
                    if s["end"] >= lo and s["start"] <= hi).strip()


def system_prompt(topic):
    subject = topic or "the product or topic shown in this video"
    return (
        f"You are analyzing frames from a video about {subject}. For the given "
        "screenshot and the transcript spoken around that moment, extract the key "
        f"takeaways about {subject} (capabilities, features, architecture, results, "
        "differentiators, or key points). Ignore generic filler. Respond ONLY with "
        'JSON: {"relevant": <true/false>, "title": "<short label of what this frame '
        'shows>", "bullets": ["<concise takeaway>", ...]} with 3-5 bullets. Set '
        '"relevant" to false (and "bullets" to []) if the frame and transcript at '
        f"this moment show no content relevant to {subject}."
    )


def analyze_frame(client, model, sys_prompt, frame, snippet, out, force):
    adir = out / "analysis"
    adir.mkdir(exist_ok=True)
    cache = adir / (frame["path"].stem + ".json")
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    b64 = base64.b64encode(frame["path"].read_bytes()).decode()
    prompt = (f"Timestamp {fmt_mmss(frame['t'])}.\n"
              f"Transcript around this moment:\n\"\"\"{snippet or '(no speech)'}\"\"\"")
    text = _vision(client, model, sys_prompt, prompt, f"data:image/jpeg;base64,{b64}")
    result = _parse_json(text)
    cache.write_text(json.dumps(result, indent=2))
    return result


def _vision(client, model, system, prompt, image_url):
    """Responses API is primary (some restricted keys only allow /v1/responses);
    chat.completions is the fallback."""
    user = [{"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": image_url}]
    try:
        resp = client.responses.create(
            model=model, instructions=system,
            input=[{"role": "user", "content": user}])
        return resp.output_text
    except Exception as e:  # noqa: BLE001 - fall back to chat.completions
        sys.stderr.write(f"      responses API failed ({e}); trying chat.completions\n")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": [
                          {"type": "text", "text": prompt},
                          {"type": "image_url", "image_url": {"url": image_url}}]}],
            response_format={"type": "json_object"})
        return resp.choices[0].message.content


def _parse_json(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {"title": "(unparsed response)", "bullets": [str(text)[:500]]}


# --- Stage 6: write report data -------------------------------------------
def build_report_json(entries, out, title, topic, model):
    """Write report.json — the structured data the web app renders server-side.

    Every frame is included, even ones the model marked relevant=false;
    filtering (like all presentation) is a render-time choice, so display
    rules can change retroactively without re-running the pipeline.
    """
    print("[6/6] writing report.json ...")
    data = {
        "version": 1,
        "title": title,
        "topic": topic,
        "model": model,
        "frames": [
            {
                "t": e["t"],
                "image": Path(e["path"]).resolve().relative_to(out.resolve()).as_posix(),
                "snippet": e["snippet"],
                "analysis": e["analysis"],  # model JSON verbatim — schema stays fluid
            }
            for e in entries
        ],
    }
    report = out / "report.json"
    report.write_text(json.dumps(data, indent=2))
    print(f"      wrote {report} ({len(entries)} frames)")
    return report


# --- Orchestrate ----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Video -> keyframe+takeaways report data.")
    ap.add_argument("video", help="input video file")
    ap.add_argument("--out", help="output dir (default: outputs/<video_stem>_analysis "
                                   "inside the skill folder)")
    ap.add_argument("--topic", help="subject the video is about, e.g. \"Acme's product\" "
                                    "(sharpens the takeaway prompt)")
    ap.add_argument("--title", help="report title (default: derived from --topic/filename)")
    ap.add_argument("--model", default="gpt-5.5", help="vision model (default: gpt-5.5)")
    ap.add_argument("--transcribe-model", default="whisper-1")
    ap.add_argument("--scene-threshold", type=float, default=0.05,
                    help="ffmpeg scene sensitivity; lower = more frames (default 0.05)")
    ap.add_argument("--interval", type=int, default=75,
                    help="coverage-floor sampling seconds; 0 disables (default 75)")
    ap.add_argument("--min-gap", type=float, default=12.0,
                    help="dedupe: min seconds between kept frames (default 12)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap analyzed frames (0 = no cap)")
    ap.add_argument("--win-before", type=float, default=8.0)
    ap.add_argument("--win-after", type=float, default=20.0)
    ap.add_argument("--skip-analysis", action="store_true",
                    help="skip stage 5 (LLM frame analysis); report.json gets "
                         "analysis: null per frame, fill in later by re-running "
                         "without this flag (stages 1-4 are cached)")
    ap.add_argument("--force", action="store_true", help="ignore caches; recompute all stages")
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        sys.exit(f"Video not found: {video}")
    out = (Path(args.out).expanduser() if args.out
           else SKILL_DIR / "outputs" / f"{video.stem}_analysis")
    out.mkdir(parents=True, exist_ok=True)
    title = args.title or (f"{args.topic} — Video Analysis" if args.topic
                           else f"{video.stem} — Video Analysis")

    client = make_client()

    audio = extract_audio(video, out, args.force)
    transcript = transcribe(client, audio, out, args.transcribe_model, args.force)
    timestamps = select_timestamps(video, out, args.scene_threshold, args.interval,
                                   args.min_gap, args.max_frames, args.force)
    frames = extract_keyframes(video, timestamps, out, args.force)

    entries = []
    if args.skip_analysis:
        print("[5/6] skipping frame analysis (--skip-analysis)")
        for frame in frames:
            snippet = snippet_for(frame["t"], transcript["segments"],
                                  args.win_before, args.win_after)
            entries.append({"t": frame["t"], "path": str(frame["path"]),
                            "snippet": snippet, "analysis": None})
    else:
        sys_prompt = system_prompt(args.topic)
        print(f"[5/6] analyzing {len(frames)} keyframes with {args.model} ...")
        skipped = 0
        for i, frame in enumerate(frames, 1):
            snippet = snippet_for(frame["t"], transcript["segments"],
                                  args.win_before, args.win_after)
            analysis = analyze_frame(client, args.model, sys_prompt, frame, snippet,
                                     out, args.force)
            entries.append({"t": frame["t"], "path": str(frame["path"]),
                            "snippet": snippet, "analysis": analysis})
            if not analysis.get("relevant", True):
                skipped += 1
                print(f"      [{i}/{len(frames)}] {fmt_mmss(frame['t'])} - "
                      "not relevant (kept in data, hidden at render time)")
                continue
            print(f"      [{i}/{len(frames)}] {fmt_mmss(frame['t'])} - "
                  f"{str(analysis.get('title', ''))[:60]}")
        if skipped:
            print(f"      {skipped} frame(s) marked not relevant (hidden at render time)")

    report = build_report_json(entries, out, title, args.topic, args.model)
    print(f"\nDone. Report data: {report} (rendered by the web app)")


if __name__ == "__main__":
    main()
