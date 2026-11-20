FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# yt-dlp needs a JS runtime (deno) to solve YouTube's player challenges;
# without it, it falls back to clients whose stream URLs 403.
COPY --from=denoland/deno:bin /deno /usr/local/bin/deno
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py jobs.py ./
COPY scripts/ scripts/
COPY templates/ templates/
COPY static/ static/
ENV DATA_DIR=/data
VOLUME /data
EXPOSE 5177
CMD ["waitress-serve", "--host", "0.0.0.0", "--port", "5177", "--threads", "8", "app:app"]
