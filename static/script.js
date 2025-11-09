const form = document.getElementById("uploadForm");
const status = document.getElementById("status");
const resultDiv = document.getElementById("result");
const historyDiv = document.getElementById("history");
const historyBtn = document.getElementById("historyBtn");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  status.textContent = "Uploading...";

  const data = new FormData(form);
  const btn = document.getElementById("submitBtn");
  btn.disabled = true;

  try {
    const res = await fetch("/upload", { method: "POST", body: data });
    const json = await res.json();
    if (!res.ok) {
      status.textContent = json.error || "Upload failed";
      resultDiv.innerHTML = "";
      btn.disabled = false;
      return;
    }

    status.textContent = `Saved. Source: ${json.source}`;
    const out = `
      <div class="small">Name: <strong>${json.name}</strong></div>
      <div class="small">Filename: <a href="/uploads/${json.filename}" target="_blank">${json.filename}</a></div>
      <h3>Recognized (best-effort):</h3>
      <pre>${json.latex || "(none)"}</pre>
      <h3>Solution:</h3>
      <pre>${json.solution || "(none)"}</pre>
    `;
    resultDiv.innerHTML = out;
  } catch (err) {
    status.textContent = "Request failed";
    resultDiv.innerHTML = "";
    console.error(err);
  } finally {
    btn.disabled = false;
  }
});

historyBtn.addEventListener("click", async () => {
  historyDiv.innerHTML = "Loading history...";
  try {
    const res = await fetch("/history");
    const arr = await res.json();
    if (!Array.isArray(arr)) {
      historyDiv.innerHTML = "No history.";
      return;
    }
    const items = arr.map(doc => {
      return `<div class="result small">
        <div><strong>${doc.name}</strong> — ${new Date(doc.createdAt).toLocaleString()}</div>
        <div>File: <a href="/uploads/${doc.filename}" target="_blank">${doc.filename}</a></div>
        <div>Latex: <pre>${doc.latex || "(none)"}</pre></div>
        <div>Solution: <pre>${doc.solution || "(none)"}</pre></div>
      </div>`;
    }).join("");
    historyDiv.innerHTML = items;
  } catch (err) {
    historyDiv.innerHTML = "Failed to load history.";
  }
});
