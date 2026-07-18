# video-analysis

Turn any video from YouTube, Facebook, any website, or your local machine, into a readable report like this:

<img width="1358" height="1063" alt="image" src="https://github.com/user-attachments/assets/2ec0ae02-b0ff-49b3-b452-e0a2f6527e5c" />


## Web app

A small Flask app wraps the pipeline: paste any video URL yt-dlp supports (YouTube,
Facebook, ...) or upload a local video file, pick a Low/Medium/High detail level, and it
downloads the video, runs the analysis in the background, and lists every report on the
home page. New reports are
transcript-first (keyframes + raw transcript, no LLM cost); flipping the report page's
switch to "AI summary" generates the per-frame takeaways on demand.

```bash
# Docker (recommended)
docker run -d -p 5177:5177 -e OPENAI_API_KEY=sk-... -v video_data:/data \
  akshaykannan/video-analysis

# Local
pip install -r requirements.txt   # plus ffmpeg/ffprobe on PATH
OPENAI_API_KEY=sk-... waitress-serve --port 5177 app:app
```

Then open http://localhost:5177. Job data (videos, transcripts, reports) lives under
`/data` in the container (`./data/` locally) — mount a volume to persist it. Interrupted
analyses resume automatically on restart thanks to the pipeline's per-stage caching.
Note: yt-dlp goes stale against YouTube over time — rebuild the image periodically.

## CLI usage

```bash
uv run --with openai python scripts/analyze_video.py INPUT.mp4 --topic "Acme's product"
```

Requires `ffmpeg`/`ffprobe` on PATH and an OpenAI key in `OPENAI_KEY` or `OPENAI_API_KEY`.
Output lands in `outputs/INPUT_analysis/report.json` (plus keyframes and transcript);
view it through the web app, which renders the data with `templates/report.html`.
Pass `--skip-analysis` to skip the LLM frame analysis (transcript-only report); rerun
without it later to fill the summaries in — earlier stages are cached.

See [SKILL.md](SKILL.md) for the full pipeline, tuning knobs, and gotchas.
