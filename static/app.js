const form = document.getElementById("analyze-form");
const urlInput = document.getElementById("url");
const fileInput = document.getElementById("file");
const detailSelect = document.getElementById("detail");
const generateBtn = document.getElementById("generate");
const formError = document.getElementById("form-error");
const grid = document.getElementById("videos");
const empty = document.getElementById("empty");

const POLL_MS = 3000;
let pollTimer = null;

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  formError.hidden = true;
  const url = urlInput.value.trim();
  const file = fileInput.files[0];
  if (!url && !file) {
    formError.textContent = "Enter a URL or choose a file.";
    formError.hidden = false;
    return;
  }
  generateBtn.disabled = true;
  try {
    await submitJob(url, detailSelect.value, file);
    urlInput.value = "";
    fileInput.value = "";
  } catch (err) {
    formError.textContent = err.message;
    formError.hidden = false;
  } finally {
    generateBtn.disabled = false;
  }
});

async function submitJob(url, detail, file) {
  let resp;
  if (file) {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("detail", detail);
    resp = await fetch("/analyze", { method: "POST", body: fd });
  } else {
    resp = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, detail }),
    });
  }
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || "Something went wrong.");
  await refresh();
  startPolling();
}

async function refresh() {
  const resp = await fetch("/api/videos");
  if (!resp.ok) return;
  const videos = await resp.json();
  render(videos);
  const active = videos.some((v) => v.status !== "done" && v.status !== "failed");
  if (!active) stopPolling();
}

function render(videos) {
  empty.hidden = videos.length > 0;
  grid.replaceChildren(...videos.map(card));
}

function card(v) {
  const done = v.status === "done";
  const failed = v.status === "failed";
  const name = v.title || v.url;

  const el = document.createElement("article");
  el.className = "card" + (done || failed ? "" : " pending");

  const wrap = document.createElement("div");
  wrap.className = "thumb-wrap";
  const img = document.createElement("img");
  // Switch to the real thumbnail only once it exists; until then point at a
  // genuinely different URL so polling swaps it in (and no cached /thumb 302
  // redirect can pin the card to the placeholder).
  img.src = v.has_thumb ? `/thumb/${v.id}` : "/static/placeholder.svg";
  img.loading = "lazy";
  img.alt = "";
  img.onerror = () => { img.src = "/static/placeholder.svg"; };
  wrap.appendChild(img);

  if (!done && !failed) {
    const overlay = document.createElement("div");
    overlay.className = "overlay";
    const label = document.createElement("span");
    label.setAttribute("aria-busy", "true");
    label.textContent = v.stage || v.status || "queued";
    overlay.append(label);
    wrap.appendChild(overlay);
  }

  const del = document.createElement("button");
  del.className = "delete";
  del.type = "button";
  del.title = "Delete";
  del.setAttribute("aria-label", "Delete video");
  del.textContent = "×";
  del.addEventListener("click", async (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!confirm(`Delete "${name}"? This permanently removes the video, frames, and report.`)) return;
    del.disabled = true;
    try {
      const resp = await fetch(`/api/videos/${v.id}`, { method: "DELETE" });
      if (!resp.ok) throw new Error();
      el.remove();
      if (!grid.children.length) empty.hidden = false;
    } catch {
      del.disabled = false;
      alert("Failed to delete.");
    }
  });
  wrap.appendChild(del);

  el.appendChild(wrap);

  const body = document.createElement("div");
  body.className = "card-body";

  const title = document.createElement(done ? "a" : "span");
  title.className = "card-title";
  if (done) title.href = `/video/${v.id}`;
  title.textContent = name;
  title.title = name;
  body.appendChild(title);

  const sub = document.createElement("div");
  sub.className = "card-sub";
  const badge = document.createElement("span");
  badge.className = "badge" + (failed ? " failed" : "");
  badge.textContent = failed ? "failed" : `${v.detail} detail`;
  sub.appendChild(badge);
  if (v.created_at) {
    const date = document.createElement("span");
    date.textContent = new Date(v.created_at * 1000).toLocaleDateString();
    sub.appendChild(date);
  }
  body.appendChild(sub);

  if (failed) {
    if (v.error) {
      const err = document.createElement("div");
      err.className = "card-sub";
      err.textContent = v.error;
      body.appendChild(err);
    }
    const row = document.createElement("div");
    row.className = "card-sub";
    if (v.url) {
      const retry = document.createElement("button");
      retry.className = "retry secondary";
      retry.textContent = "Retry";
      retry.addEventListener("click", async (e) => {
        e.preventDefault();
        retry.disabled = true;
        try { await submitJob(v.url, v.detail); } catch { retry.disabled = false; }
      });
      row.appendChild(retry);
    }
    const log = document.createElement("a");
    log.href = `/log/${v.id}`;
    log.target = "_blank";
    log.textContent = "view log";
    row.appendChild(log);
    body.appendChild(row);
  }

  el.appendChild(body);
  return el;
}

function startPolling() {
  if (!pollTimer) pollTimer = setInterval(refresh, POLL_MS);
}

function stopPolling() {
  clearInterval(pollTimer);
  pollTimer = null;
}

refresh();
startPolling();
