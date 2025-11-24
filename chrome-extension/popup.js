//const API_URL = "http://127.0.0.1:8000/predict_batch";
const API_URL = "https://sarabelhouari-youtube-sentiment-api.hf.space/predict_batch";
const analyzeBtn = document.getElementById("analyzeBtn");
const copyBtn = document.getElementById("copyBtn");
const statusEl = document.getElementById("status");

const statsSection = document.getElementById("statsSection");
const filtersSection = document.getElementById("filtersSection");
const listSection = document.getElementById("listSection");
const resultsList = document.getElementById("resultsList");

const posPct = document.getElementById("posPct");
const neuPct = document.getElementById("neuPct");
const negPct = document.getElementById("negPct");

let allResults = [];
let currentFilter = "all";

// --- Dark mode persistence ---
const darkToggle = document.getElementById("darkToggle");
chrome.storage.local.get(["darkMode"], (res) => {
  if (res.darkMode) {
    document.body.classList.add("dark");
    darkToggle.checked = true;
  }
});
darkToggle.addEventListener("change", () => {
  document.body.classList.toggle("dark");
  chrome.storage.local.set({ darkMode: darkToggle.checked });
});

// --- UI helpers ---
function setStatus(msg, type="info") {
  statusEl.textContent = msg;
  statusEl.classList.remove("hidden");
  statusEl.style.borderLeft = type === "error" ? "4px solid #dc2626" : "4px solid #2563eb";
}
function clearStatus() {
  statusEl.classList.add("hidden");
}

// Map label->badge class/text
function labelInfo(label) {
  if (label === 1) return { cls: "pos", txt: "Positif" };
  if (label === 0) return { cls: "neu", txt: "Neutre" };
  return { cls: "neg", txt: "Négatif" };
}

// Render list with filter
function renderResults() {
  resultsList.innerHTML = "";

  const filtered = currentFilter === "all"
    ? allResults
    : allResults.filter(r => String(r.label) === currentFilter);

  filtered.forEach(r => {
    const info = labelInfo(r.label);
    const conf = Math.max(r.probabilities["-1"], r.probabilities["0"], r.probabilities["1"]);
    const li = document.createElement("li");
    li.className = "result";
    li.innerHTML = `
      <span class="badge ${info.cls}">${info.txt}</span>
      <b>${(conf * 100).toFixed(1)}%</b>
      <div>${r.text}</div>
    `;
    resultsList.appendChild(li);
  });
}

// Filters click
document.querySelectorAll(".filter").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filter").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentFilter = btn.dataset.filter;
    renderResults();
  });
});

// Copy results
copyBtn.addEventListener("click", async () => {
  const txt = allResults.map(r => `${r.label}\t${r.text}`).join("\n");
  await navigator.clipboard.writeText(txt);
  setStatus("Résultats copiés ");
});

// Main analysis
analyzeBtn.addEventListener("click", async () => {
  clearStatus();
  analyzeBtn.disabled = true;
  copyBtn.disabled = true;
  setStatus("Extraction des commentaires...");

  // Demander les commentaires au content script
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const comments = await new Promise((resolve) => {
    chrome.tabs.sendMessage(tab.id, { type: "GET_COMMENTS" }, (res) => resolve(res?.comments || []));
  });

  if (!comments.length) {
    setStatus("Aucun commentaire visible détecté. Scrolle un peu.", "error");
    analyzeBtn.disabled = false;
    return;
  }

  setStatus(`Analyse ML en cours (${comments.length} commentaires)...`);

  try {
    const resp = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts: comments })
    });

    if (!resp.ok) {
      throw new Error(`API error ${resp.status}`);
    }

    const data = await resp.json();
    allResults = data.results;

    // Stats globales
    posPct.textContent = (data.stats["1"] * 100).toFixed(1) + "%";
    neuPct.textContent = (data.stats["0"] * 100).toFixed(1) + "%";
    negPct.textContent = (data.stats["-1"] * 100).toFixed(1) + "%";

    statsSection.classList.remove("hidden");
    filtersSection.classList.remove("hidden");
    listSection.classList.remove("hidden");

    currentFilter = "all";
    document.querySelectorAll(".filter").forEach(b => b.classList.remove("active"));
    document.querySelector('[data-filter="all"]').classList.add("active");

    renderResults();

    setStatus("Analyse terminée ");
    copyBtn.disabled = false;
  } catch (e) {
    setStatus("Erreur API. Vérifie que FastAPI tourne + CORS ok.", "error");
  } finally {
    analyzeBtn.disabled = false;
  }
});
