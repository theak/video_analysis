FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# yt-dlp needs a JS runtime (deno) to solve YouTube's player challenges;
# without it, it falls back to clients whose stream URLs 403.
COPY --from=denoland/deno:bin /deno /usr/local/bin/deno
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# CI passes the exact yt-dlp version and records it as a label, so the
# scheduled workflow can skip rebuilds when yt-dlp hasn't changed.
ARG YTDLP_VERSION=""
RUN if [ -n "$YTDLP_VERSION" ]; then \
      pip install --no-cache-dir "yt-dlp[default]==$YTDLP_VERSION"; \
    fi
LABEL ytdlp.version="${YTDLP_VERSION}"
COPY app.py jobs.py ./
COPY scripts/ scripts/
COPY templates/ templates/
COPY static/ static/
ENV DATA_DIR=/data
VOLUME /data
EXPOSE 5177
CMD ["waitress-serve", "--host", "0.0.0.0", "--port", "5177", "--threads", "8", "app:app"]
