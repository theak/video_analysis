#!/usr/bin/env python3
"""
Turn any video into a self-contained HTML report: keyframes + concise LLM takeaways.

Pipeline (each stage caches to the output dir, so reruns skip completed work):
  1. Extract audio                (ffmpeg)
  2. Transcribe                   (OpenAI whisper-1, verbose_json; auto-chunks if >24MB)
  3. Select keyframe timestamps   (ffmpeg scene detection + fixed-interval coverage floor)
  4. Extract one keyframe / ts    (ffmpeg; timestamp embedded in filename)
  5. Analyze each keyframe        (vision model; image + aligned transcript -> JSON bullets)
  6. Build report.html            (screenshots base64-embedded + bullets + transcript context)

Requires: ffmpeg + ffprobe on PATH; an OpenAI key in OPENAI_KEY or OPENAI_API_KEY.

Run:
  uv run --with openai python analyze_video.py INPUT.mp4 --topic "Acme's product"
"""

import argparse
import base64
import html
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


# --- Stage 6: build HTML --------------------------------------------------
def build_html(entries, out, title, model):
    print("[6/6] building HTML ...")
    cards = []
    for e in entries:
        b64 = base64.b64encode(Path(e["path"]).read_bytes()).decode()
        bullets = "\n".join(f"      <li>{html.escape(str(b))}</li>"
                            for b in e["analysis"].get("bullets", []))
        h_title = html.escape(str(e["analysis"].get("title", "")))
        snippet = html.escape(e["snippet"] or "(no speech)")
        cards.append(f"""  <section class="card">
    <div class="ts">{fmt_mmss(e['t'])}</div>
    <button class="del" type="button" title="Delete card" aria-label="Delete card">&times;</button>
    <img src="data:image/jpeg;base64,{b64}" alt="keyframe at {fmt_mmss(e['t'])}">
    <div class="body">
      <h2>{h_title}</h2>
      <ul>
{bullets}
      </ul>
      <details><summary>Transcript context</summary><p>{snippet}</p></details>
    </div>
  </section>""")

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 16px/1.5 -apple-system, system-ui, sans-serif; margin: 0;
         background: #0f1115; color: #e7e9ee; }}
  header {{ padding: 32px 24px; border-bottom: 1px solid #2a2e39; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; }}
  header p {{ margin: 0; color: #9aa1ad; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
  .card {{ display: grid; grid-template-columns: 480px 1fr; gap: 20px;
          background: #171a21; border: 1px solid #2a2e39; border-radius: 12px;
          padding: 16px; margin-bottom: 24px; position: relative; }}
  .card img {{ width: 100%; border-radius: 8px; display: block; cursor: zoom-in; }}
  #lightbox {{ position: fixed; inset: 0; z-index: 100; display: none;
             background: #000; cursor: zoom-out; }}
  #lightbox.open {{ display: block; }}
  #lightbox img {{ width: 100vw; height: 100vh; object-fit: contain; }}
  #lightbox .hint {{ position: fixed; top: 16px; right: 20px; color: #fff9;
                   font-size: 13px; }}
  .ts {{ position: absolute; top: 24px; left: 24px; background: #000a;
        color: #fff; font-variant-numeric: tabular-nums; font-size: 13px;
        padding: 2px 8px; border-radius: 6px; }}
  .del {{ position: absolute; top: 20px; right: 20px; z-index: 3; width: 26px;
         height: 26px; padding: 0; border: none; border-radius: 50%;
         background: #000a; color: #fff; font: 18px/1 system-ui, sans-serif;
         cursor: pointer; display: flex; align-items: center;
         justify-content: center; transition: background .15s; }}
  .del:hover {{ background: #e5484d; }}
  .body h2 {{ margin: 0 0 10px; padding-right: 34px; font-size: 18px; color: #fff; }}
  .body ul {{ margin: 0 0 12px; padding-left: 20px; }}
  .body li {{ margin-bottom: 6px; }}
  details summary {{ cursor: pointer; color: #9aa1ad; font-size: 14px; }}
  details p {{ color: #9aa1ad; font-size: 14px; }}
  @media screen and (max-width: 820px) {{ .card {{ grid-template-columns: 1fr; }} }}
  @media print {{
    @page {{ margin: 1.2cm; }}
    :root {{ color-scheme: light; }}
    body {{ background: #fff; color: #000; font-size: 11px; }}
    header {{ padding: 0 0 10px; border-bottom: 1px solid #000; }}
    header h1 {{ color: #000; font-size: 20px; }}
    header p {{ color: #333; }}
    main {{ max-width: none; margin: 0; padding: 0; }}
    /* Keep the screen's image-left/text-right layout but shrink the image so
       several cards fit per page; force light colors regardless of the
       browser's "print background graphics" setting. */
    .card {{ grid-template-columns: 300px 1fr; gap: 14px; background: #fff;
            border: 1px solid #bbb; border-radius: 6px; padding: 10px;
            margin-bottom: 12px; break-inside: avoid; page-break-inside: avoid; }}
    .card img {{ border: 1px solid #ddd; }}
    .ts {{ top: 16px; left: 16px; }}
    .body h2 {{ color: #000; font-size: 14px; margin: 0 0 6px; }}
    .body ul {{ margin: 0; }}
    .body li {{ margin-bottom: 3px; }}
    details {{ display: none; }}  /* collapsed transcript is dead weight on paper */
    .del {{ display: none; }}     /* no delete buttons on paper */
  }}
</style></head><body>
<header>
  <h1>{html.escape(title)}</h1>
  <p>{len(entries)} keyframes · analyzed with {html.escape(model)}</p>
</header>
<main>
{chr(10).join(cards)}
</main>
<div id="lightbox"><span class="hint">&larr; / &rarr; navigate &middot; click or Esc to close</span><img alt="fullscreen keyframe"></div>
<script>
  const lb = document.getElementById('lightbox');
  const lbImg = lb.querySelector('img');
  let current = -1;
  const imgList = () => Array.from(document.querySelectorAll('.card img'));  // live, survives deletes
  const show = i => {{
    const list = imgList();
    if (!list.length) return;
    current = (i + list.length) % list.length;   // wrap at both ends
    lbImg.src = list[current].src;
    lb.classList.add('open');
  }};
  // Client-side delete: removes the card from the DOM only, so a reload restores it.
  document.querySelectorAll('.card .del').forEach(btn =>
    btn.addEventListener('click', e => {{ e.stopPropagation(); btn.closest('.card').remove(); }}));
  document.querySelectorAll('.card img').forEach(img =>
    img.addEventListener('click', () => show(imgList().indexOf(img))));
  const close = () => {{ lb.classList.remove('open'); lbImg.removeAttribute('src'); current = -1; }};
  lb.addEventListener('click', close);
  document.addEventListener('keydown', e => {{
    if (!lb.classList.contains('open')) return;
    if (e.key === 'Escape') close();
    else if (e.key === 'ArrowRight') {{ e.preventDefault(); show(current + 1); }}
    else if (e.key === 'ArrowLeft')  {{ e.preventDefault(); show(current - 1); }}
  }});
</script>
</body></html>"""

    report = out / "report.html"
    report.write_text(doc)
    print(f"      wrote {report}")
    return report


# --- Orchestrate ----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Video -> keyframe+takeaways HTML report.")
    ap.add_argument("video", help="input video file")
    ap.add_argument("--out", help="output dir (default: outputs/<video_stem>_analysis "
                                   "inside the skill folder)")
    ap.add_argument("--topic", help="subject the video is about, e.g. \"Acme's product\" "
                                    "(sharpens the takeaway prompt)")
    ap.add_argument("--title", help="HTML report title (default: derived from --topic/filename)")
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

    sys_prompt = system_prompt(args.topic)
    print(f"[5/6] analyzing {len(frames)} keyframes with {args.model} ...")
    entries = []
    skipped = 0
    for i, frame in enumerate(frames, 1):
        snippet = snippet_for(frame["t"], transcript["segments"],
                              args.win_before, args.win_after)
        analysis = analyze_frame(client, args.model, sys_prompt, frame, snippet,
                                 out, args.force)
        if not analysis.get("relevant", True):
            skipped += 1
            print(f"      [{i}/{len(frames)}] {fmt_mmss(frame['t'])} - "
                  "skipped (not relevant)")
            continue
        entries.append({"t": frame["t"], "path": str(frame["path"]),
                        "snippet": snippet, "analysis": analysis})
        print(f"      [{i}/{len(frames)}] {fmt_mmss(frame['t'])} - "
              f"{str(analysis.get('title', ''))[:60]}")
    if skipped:
        print(f"      skipped {skipped} frame(s) with no relevant content")

    report = build_html(entries, out, title, args.model)
    print(f"\nDone. Open: {report}")


if __name__ == "__main__":
    main()
