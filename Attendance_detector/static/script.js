"use strict";

const $ = (id) => document.getElementById(id);

const fileInput = $("fileInput");
const dropZone = $("dropZone");
const binarize = $("binarize");
const previewBtn = $("previewBtn");
const analyzeBtn = $("analyzeBtn");
const resetBtn = $("resetBtn");

const imagesPanel = $("imagesPanel");
const originalImg = $("originalImg");
const processedImg = $("processedImg");
const processedPlaceholder = $("processedPlaceholder");

const resultsPanel = $("resultsPanel");
const statsEl = $("stats");
const resultsTable = $("resultsTable");
const noMatch = $("noMatch");
const modelName = $("modelName");
const searchInput = $("searchInput");
const copyBtn = $("copyBtn");
const csvBtn = $("csvBtn");

const overlay = $("overlay");
const overlayText = $("overlayText");
const overlaySub = $("overlaySub");
const toasts = $("toasts");
const steps = $("steps");

let selectedFile = null;
let lastResult = null;          // { dates, students }
let sortDir = 0;                // 0 none, 1 asc, -1 desc (by name)

/* ====================== file selection ====================== */
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

["dragenter", "dragover"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add("is-drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove("is-drag"); })
);
dropZone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) setFile(f);
});

function setFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    toast("Please choose an image file (JPG, PNG or WEBP).", "error");
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    toast("That image is larger than 16 MB. Please use a smaller file.", "error");
    return;
  }
  selectedFile = file;
  previewBtn.disabled = false;
  analyzeBtn.disabled = false;
  resetBtn.hidden = false;

  resultsPanel.hidden = true;
  lastResult = null;

  const reader = new FileReader();
  reader.onload = (e) => {
    originalImg.src = e.target.result;
    processedImg.hidden = true;
    processedImg.removeAttribute("src");
    processedPlaceholder.hidden = false;
    imagesPanel.hidden = false;
    setStep(2);
    imagesPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };
  reader.readAsDataURL(file);
}

resetBtn.addEventListener("click", () => {
  selectedFile = null;
  lastResult = null;
  fileInput.value = "";
  previewBtn.disabled = true;
  analyzeBtn.disabled = true;
  resetBtn.hidden = true;
  imagesPanel.hidden = true;
  resultsPanel.hidden = true;
  setStep(1);
  window.scrollTo({ top: 0, behavior: "smooth" });
});

/* ====================== sample images ====================== */
document.querySelectorAll(".sample").forEach((btn) =>
  btn.addEventListener("click", () => loadSample(btn.dataset.src))
);

async function loadSample(url) {
  if (!url) return;
  showOverlay("Loading example…");
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error("fetch failed");
    const blob = await res.blob();
    const name = url.split("/").pop() || "sample.png";
    const file = new File([blob], name, { type: blob.type || "image/png" });
    setFile(file);
    toast("Example loaded — click Analyze attendance.", "info");
  } catch {
    toast("Could not load the example image.", "error");
  } finally {
    hideOverlay();
  }
}

/* ====================== requests ====================== */
function buildForm() {
  const fd = new FormData();
  fd.append("image", selectedFile);
  fd.append("binarize", binarize.checked ? "1" : "0");
  return fd;
}

function setBusy(busy) {
  previewBtn.disabled = busy || !selectedFile;
  analyzeBtn.disabled = busy || !selectedFile;
  resetBtn.disabled = busy;
}

function showOverlay(text, sub = "") {
  overlayText.textContent = text;
  overlaySub.textContent = sub;
  overlay.hidden = false;
}
function hideOverlay() { overlay.hidden = true; }

previewBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  setBusy(true);
  showOverlay("Applying filters…", "Cleaning up the image");
  try {
    const res = await fetch("/api/preview", { method: "POST", body: buildForm() });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Preview failed.");
    showProcessed(data.processed_image);
    toast("Filter applied — review the enhanced image, then analyze.", "info");
  } catch (err) {
    toast(err.message, "error");
  } finally {
    hideOverlay();
    setBusy(false);
  }
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  setBusy(true);
  showOverlay("Reading the sheet…", "The AI is checking each circle (10–30s)");
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: buildForm() });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Analysis failed.");
    showProcessed(data.processed_image);
    lastResult = normalizeResult(data.result);
    modelName.textContent = data.model || "—";
    renderAll();
    setStep(3, true);
    toast("Done! Attendance extracted.", "success");
    resultsPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    toast(err.message, "error");
  } finally {
    hideOverlay();
    setBusy(false);
  }
});

function showProcessed(dataUri) {
  processedImg.src = dataUri;
  processedImg.hidden = false;
  processedPlaceholder.hidden = true;
  imagesPanel.hidden = false;
}

/* ====================== results ====================== */
function normalizeResult(result) {
  if (!result || !Array.isArray(result.students)) return null;
  const dates =
    Array.isArray(result.dates) && result.dates.length ? result.dates.map(String) : ["attendance"];
  const students = result.students.map((s) => ({
    name: s && s.name ? String(s.name) : "—",
    attendance: (s && s.attendance) || {},
  }));
  return { dates, students };
}

function statusOf(student, date) {
  const v = (student.attendance[date] || "unclear").toString().toLowerCase();
  if (v === "present" || v === "absent") return v;
  return "unclear";
}

function renderAll() {
  if (!lastResult || lastResult.students.length === 0) {
    toast("No student rows were found. Try a clearer image or high-contrast mode.", "error");
    return;
  }
  renderStats();
  renderTable();
  resultsPanel.hidden = false;
}

function renderStats() {
  const { dates, students } = lastResult;
  let present = 0, absent = 0;
  students.forEach((s) => dates.forEach((d) => {
    const st = statusOf(s, d);
    if (st === "present") present++;
    else if (st === "absent") absent++;
  }));
  const marked = present + absent;
  const rate = marked ? Math.round((present / marked) * 100) : 0;

  statsEl.innerHTML = `
    <div class="stat"><span class="stat-value">${students.length}</span><span class="stat-label">Students</span></div>
    <div class="stat present"><span class="stat-value">${present}</span><span class="stat-label">Present</span></div>
    <div class="stat absent"><span class="stat-value">${absent}</span><span class="stat-label">Absent</span></div>
    <div class="stat rate"><span class="stat-value">${rate}%</span><span class="stat-label">Attendance</span></div>`;
}

function renderTable() {
  const { dates } = lastResult;
  const query = (searchInput.value || "").trim().toLowerCase();

  let rows = lastResult.students
    .map((s, i) => ({ s, i }))
    .filter(({ s }) => s.name.toLowerCase().includes(query));

  if (sortDir !== 0) {
    rows.sort((a, b) => sortDir * a.s.name.localeCompare(b.s.name));
  }

  const ind = sortDir === 0 ? "" : sortDir === 1 ? "▲" : "▼";
  let html = `<thead><tr><th class="idx">#</th>
    <th class="sortable" id="nameHead">Student <span class="sort-ind">${ind}</span></th>`;
  dates.forEach((d) => (html += `<th>${escapeHtml(d)}</th>`));
  html += "</tr></thead><tbody>";

  rows.forEach(({ s }, n) => {
    html += `<tr><td class="idx">${n + 1}</td><td class="name-cell">${escapeHtml(s.name)}</td>`;
    dates.forEach((d) => {
      const st = statusOf(s, d);
      const label = st === "present" ? "Present" : st === "absent" ? "Absent" : "Unclear";
      html += `<td><span class="badge ${st}">${label}</span></td>`;
    });
    html += "</tr>";
  });
  html += "</tbody>";
  resultsTable.innerHTML = html;

  resultsTable.hidden = rows.length === 0;
  noMatch.hidden = rows.length !== 0;

  const nameHead = $("nameHead");
  if (nameHead) nameHead.addEventListener("click", () => {
    sortDir = sortDir === 1 ? -1 : 1;
    renderTable();
  });
}

searchInput && searchInput.addEventListener("input", renderTable);

/* ====================== export ====================== */
function toRows() {
  const { dates, students } = lastResult;
  const header = ["Student", ...dates];
  const body = students.map((s) => [s.name, ...dates.map((d) => statusOf(s, d))]);
  return [header, ...body];
}

function toCSV() {
  return toRows()
    .map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(","))
    .join("\n");
}

csvBtn.addEventListener("click", () => {
  if (!lastResult) return;
  const blob = new Blob([toCSV()], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "attendance.csv";
  a.click();
  URL.revokeObjectURL(url);
  toast("CSV downloaded.", "success");
});

copyBtn.addEventListener("click", async () => {
  if (!lastResult) return;
  const tsv = toRows().map((r) => r.join("\t")).join("\n");
  try {
    await navigator.clipboard.writeText(tsv);
    toast("Copied to clipboard — paste into Excel/Sheets.", "success");
  } catch {
    toast("Could not access the clipboard.", "error");
  }
});

/* ====================== stepper ====================== */
function setStep(n, doneAll = false) {
  [...steps.children].forEach((li) => {
    const step = Number(li.dataset.step);
    li.classList.toggle("is-active", step === n);
    li.classList.toggle("is-done", doneAll ? step < n : step < n);
  });
}

/* ====================== toasts ====================== */
const ICONS = {
  success: '<path d="M20 6L9 17l-5-5"/>',
  error: '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
  info: '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
};

function toast(message, type = "info") {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.innerHTML =
    `<svg class="toast-icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[type] || ICONS.info}</svg>` +
    `<span>${escapeHtml(message)}</span>`;
  toasts.appendChild(el);
  setTimeout(() => {
    el.classList.add("out");
    el.addEventListener("animationend", () => el.remove(), { once: true });
  }, 4200);
}

/* ====================== utils ====================== */
function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}
