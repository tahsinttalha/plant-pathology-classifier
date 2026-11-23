/**
 * Plant Pathology Classifier — Minimalist Composer & Report Controller
 * Manages landing -> chatbot-style report transition, Upload / Drag & Drop /
 * Camera Capture / Google Drive import, and toggleable Threshold Gate slider.
 */

(() => {
  "use strict";

  const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

  const el = {
    body: document.body,
    heroHeader: document.getElementById("hero-header"),
    reportStage: document.getElementById("report-stage"),

    centerBox: document.getElementById("center-box"),
    dropPrompt: document.getElementById("drop-prompt"),
    dropPromptText: document.getElementById("drop-prompt-text"),
    fileInput: document.getElementById("file-input"),
    cameraInput: document.getElementById("camera-input"),

    btnUpload: document.getElementById("btn-upload"),
    btnDrop: document.getElementById("btn-drop"),
    btnCamera: document.getElementById("btn-camera"),
    btnDrive: document.getElementById("btn-drive"),
    btnThresholdToggle: document.getElementById("btn-threshold-toggle"),

    drivePopover: document.getElementById("drive-popover"),
    driveCloseBtn: document.getElementById("drive-close-btn"),
    driveForm: document.getElementById("drive-form"),
    driveUrlInput: document.getElementById("drive-url-input"),

    thresholdPopover: document.getElementById("threshold-popover"),
    thresholdSlider: document.getElementById("threshold-slider"),
    thresholdBtnVal: document.getElementById("threshold-btn-val"),
    thresholdPopoverVal: document.getElementById("threshold-popover-val"),

    errorBox: document.getElementById("error-box"),
    errorLabel: document.getElementById("error-label"),
    errorMessage: document.getElementById("error-message"),
    specimenBox: document.getElementById("specimen-box"),
    previewImg: document.getElementById("preview-img"),
    specimenName: document.getElementById("specimen-name"),
    specimenMeta: document.getElementById("specimen-meta"),
    newScanBtn: document.getElementById("new-scan-btn"),

    loadingBox: document.getElementById("loading-box"),
    reportContent: document.getElementById("report-content"),
    primaryBox: document.getElementById("primary-box"),
    statusBadge: document.getElementById("status-badge"),
    confidentReadout: document.getElementById("confident-readout"),
    cropSpecies: document.getElementById("crop-species"),
    pathologyName: document.getElementById("pathology-name"),
    uncertainReadout: document.getElementById("uncertain-readout"),
    uncertainMessage: document.getElementById("uncertain-message"),
    uncertainSubnote: document.getElementById("uncertain-subnote"),

    metricConfidence: document.getElementById("metric-confidence"),
    metricGate: document.getElementById("metric-gate"),
    metricLatency: document.getElementById("metric-latency"),
    topPredictionsList: document.getElementById("top-predictions-list"),
  };

  let activeSource = null; // { kind: "file", file } | { kind: "drive", url }
  let previewObjectUrl = null;
  let activeAbortController = null;

  function parseLabel(rawLabel) {
    if (!rawLabel || typeof rawLabel !== "string") {
      return { species: "Unknown Crop", pathology: "Unknown Condition" };
    }
    const idx = rawLabel.indexOf(":");
    if (idx === -1) {
      return { species: "Plant Specimen", pathology: humanizeToken(rawLabel) };
    }
    const rawSpecies = rawLabel.slice(0, idx).trim();
    const rawPathology = rawLabel.slice(idx + 1).trim();
    const species = rawSpecies
      .replace(/\s*Leaf\b/i, "")
      .replace(/_/g, " ")
      .replace(/\(/g, " (")
      .replace(/\s+/g, " ")
      .trim();

    return {
      species: species || "Plant Specimen",
      pathology: humanizeToken(rawPathology),
    };
  }

  function humanizeToken(token) {
    if (!token) return "Unknown";
    const cleaned = token
      .replace(/_/g, " ")
      .replace(/\s+/g, " ")
      .replace(/\(\s+/g, "(")
      .replace(/\s+\)/g, ")")
      .trim();

    if (cleaned.toLowerCase() === "healthy") {
      return "Healthy (No Disease Detected)";
    }
    return cleaned.replace(/\b([a-z])/g, (m) => m.toUpperCase());
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) return "0 KB";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  function formatPercent(prob) {
    return `${(Number(prob) * 100).toFixed(2)}%`;
  }

  function closePopovers() {
    el.thresholdPopover.hidden = true;
    el.btnThresholdToggle.setAttribute("aria-expanded", "false");
    el.drivePopover.hidden = true;
    el.btnDrive.setAttribute("aria-expanded", "false");
  }

  function enterReportMode() {
    el.body.classList.remove("state-landing");
    el.body.classList.add("state-report");
    el.heroHeader.hidden = true;
    el.reportStage.hidden = false;
    el.dropPromptText.textContent =
      "Drop another leaf photograph to run a new diagnostic report";
  }

  function returnToLanding() {
    if (activeAbortController) {
      activeAbortController.abort();
      activeAbortController = null;
    }
    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
      previewObjectUrl = null;
    }

    activeSource = null;
    el.fileInput.value = "";
    el.cameraInput.value = "";
    el.driveUrlInput.value = "";
    el.previewImg.removeAttribute("src");
    el.specimenBox.hidden = true;
    el.loadingBox.hidden = true;
    el.reportContent.hidden = true;
    el.errorBox.hidden = true;
    el.reportStage.hidden = true;
    el.heroHeader.hidden = false;
    el.body.classList.remove("state-report");
    el.body.classList.add("state-landing");
    el.dropPromptText.textContent =
      "Drop a leaf photograph here, or choose an input below";
    closePopovers();
  }

  function showError(message, label = "INPUT REJECTED") {
    enterReportMode();
    el.loadingBox.hidden = true;
    el.errorLabel.textContent = label;
    el.errorMessage.textContent = message;
    el.errorBox.hidden = false;
  }

  function clearError() {
    el.errorBox.hidden = true;
    el.errorLabel.textContent = "INPUT REJECTED";
    el.errorMessage.textContent = "";
  }

  function showFilePreview(file) {
    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
    }
    previewObjectUrl = URL.createObjectURL(file);

    el.specimenName.textContent = file.name || "Captured Leaf Specimen";
    el.previewImg.onload = () => {
      const fmt = (file.type || "IMAGE").replace("image/", "").toUpperCase();
      el.specimenMeta.textContent = `${el.previewImg.naturalWidth}×${el.previewImg.naturalHeight} PX · ${fmt} · ${formatBytes(file.size)}`;
    };
    el.previewImg.src = previewObjectUrl;
    el.specimenBox.hidden = false;
  }

  function showDrivePreview(_url, dims) {
    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
      previewObjectUrl = null;
    }
    el.specimenName.textContent = "Google Drive Specimen";
    el.specimenMeta.textContent = dims
      ? `${dims} PX · GOOGLE DRIVE`
      : "GOOGLE DRIVE IMPORT";
    el.previewImg.removeAttribute("src");
    el.specimenBox.hidden = false;
  }

  function renderReport(data) {
    const isConfident = data.status === "confident";
    const topCandidate =
      Array.isArray(data.top_predictions) && data.top_predictions.length > 0
        ? data.top_predictions[0]
        : null;

    el.loadingBox.hidden = true;
    el.reportContent.hidden = false;

    el.primaryBox.className = isConfident
      ? "report-box diagnosis-card diagnosis-card--confident"
      : "report-box diagnosis-card diagnosis-card--uncertain";

    el.statusBadge.className = isConfident
      ? "status-box status-box--confident mono-tabular"
      : "status-box status-box--uncertain mono-tabular";
    el.statusBadge.textContent = isConfident ? "CONFIDENT" : "UNCERTAIN";

    if (isConfident) {
      const parsed = parseLabel(data.diagnosis);
      el.cropSpecies.textContent = parsed.species;
      el.pathologyName.textContent = parsed.pathology;
      el.confidentReadout.hidden = false;
      el.uncertainReadout.hidden = true;
    } else {
      el.confidentReadout.hidden = true;
      el.uncertainReadout.hidden = false;
      el.uncertainMessage.textContent = data.diagnosis;

      if (topCandidate) {
        const parsedTop = parseLabel(topCandidate.label);
        el.uncertainSubnote.textContent =
          `Highest sub-threshold candidate: ${parsedTop.species} — ${parsedTop.pathology} (${formatPercent(topCandidate.confidence)} vs. ${formatPercent(data.threshold)} gate).`;
      } else {
        el.uncertainSubnote.textContent = "";
      }
    }

    el.metricConfidence.textContent = formatPercent(data.confidence);
    el.metricGate.textContent = isConfident
      ? `PASSES ≥ ${formatPercent(data.threshold)} GATE`
      : `BELOW ${formatPercent(data.threshold)} SAFETY GATE`;
    el.metricLatency.textContent = `${Number(data.latency_ms).toFixed(2)} ms`;

    el.topPredictionsList.innerHTML = "";
    (data.top_predictions || []).forEach((item, index) => {
      const parsed = parseLabel(item.label);
      const pctWidth = Math.max(0, Math.min(100, Number(item.confidence) * 100));

      const li = document.createElement("li");
      li.className = "ranked-item";

      const row = document.createElement("div");
      row.className = "ranked-item__row";

      const rank = document.createElement("span");
      rank.className = "ranked-item__rank mono-tabular";
      rank.textContent = String(index + 1).padStart(2, "0");

      const labelWrap = document.createElement("span");
      labelWrap.className = "ranked-item__label";

      const speciesSpan = document.createElement("span");
      speciesSpan.className = "ranked-item__species";
      speciesSpan.textContent = `${parsed.species} / `;
      labelWrap.appendChild(speciesSpan);
      labelWrap.appendChild(document.createTextNode(parsed.pathology));

      const score = document.createElement("span");
      score.className = "ranked-item__score mono-tabular";
      score.textContent = formatPercent(item.confidence);

      row.appendChild(rank);
      row.appendChild(labelWrap);
      row.appendChild(score);

      const track = document.createElement("div");
      track.className = "hairline-track";

      const fill = document.createElement("div");
      fill.className = "hairline-fill";
      if (index === 0) {
        fill.classList.add(
          isConfident ? "hairline-fill--confident" : "hairline-fill--uncertain"
        );
      }
      fill.style.width = `${pctWidth.toFixed(2)}%`;
      track.appendChild(fill);

      li.appendChild(row);
      li.appendChild(track);
      el.topPredictionsList.appendChild(li);
    });

    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function runFileInference(file) {
    clearError();
    closePopovers();

    if (file.type && !ALLOWED_TYPES.has(file.type.toLowerCase())) {
      showError(
        `Unsupported format (${file.type}). Upload a JPEG, PNG, or WEBP image.`
      );
      return;
    }

    enterReportMode();
    showFilePreview(file);
    el.loadingBox.hidden = false;

    if (activeAbortController) {
      activeAbortController.abort();
    }
    activeAbortController = new AbortController();

    const threshold = Number(el.thresholdSlider.value).toFixed(2);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        `/predict?threshold=${encodeURIComponent(threshold)}`,
        {
          method: "POST",
          body: formData,
          signal: activeAbortController.signal,
        }
      );
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          typeof payload.detail === "string"
            ? payload.detail
            : `Prediction failed (HTTP ${response.status}).`
        );
      }
      renderReport(payload);
    } catch (err) {
      if (err && err.name === "AbortError") return;
      if (err instanceof TypeError) {
        showError(
          "Cannot reach the FastAPI inference server (/predict). Start the backend with uvicorn or check your deployment endpoint.",
          "SERVER UNREACHABLE"
        );
        return;
      }
      showError(err instanceof Error ? err.message : "Prediction failed.");
    }
  }

  async function runDriveInference(url) {
    clearError();
    closePopovers();
    enterReportMode();
    showDrivePreview(url, null);
    el.loadingBox.hidden = false;

    if (activeAbortController) {
      activeAbortController.abort();
    }
    activeAbortController = new AbortController();

    const threshold = Number(el.thresholdSlider.value).toFixed(2);

    try {
      const response = await fetch(
        `/predict/drive?threshold=${encodeURIComponent(threshold)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url }),
          signal: activeAbortController.signal,
        }
      );
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          typeof payload.detail === "string"
            ? payload.detail
            : `Google Drive import failed (HTTP ${response.status}).`
        );
      }
      if (payload.image_dimensions) {
        el.specimenMeta.textContent = `${payload.image_dimensions} PX · GOOGLE DRIVE`;
      }
      renderReport(payload);
    } catch (err) {
      if (err && err.name === "AbortError") return;
      if (err instanceof TypeError) {
        showError(
          "Cannot reach the FastAPI inference server (/predict/drive). Start the backend with uvicorn or check your deployment endpoint.",
          "SERVER UNREACHABLE"
        );
        return;
      }
      showError(
        err instanceof Error ? err.message : "Google Drive import failed."
      );
    }
  }

  function handleFileSelected(file) {
    if (!file) return;
    activeSource = { kind: "file", file };
    runFileInference(file);
  }

  // Upload & Camera & Drop interactions
  el.btnUpload.addEventListener("click", () => {
    closePopovers();
    el.fileInput.click();
  });

  el.btnDrop.addEventListener("click", () => {
    closePopovers();
    el.dropPrompt.focus();
    el.fileInput.click();
  });

  el.dropPrompt.addEventListener("click", () => {
    closePopovers();
    el.fileInput.click();
  });

  el.dropPrompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      el.fileInput.click();
    }
  });

  el.btnCamera.addEventListener("click", () => {
    closePopovers();
    el.cameraInput.click();
  });

  el.fileInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) handleFileSelected(file);
  });

  el.cameraInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) handleFileSelected(file);
  });

  ["dragenter", "dragover"].forEach((name) => {
    el.centerBox.addEventListener(name, (event) => {
      event.preventDefault();
      event.stopPropagation();
      el.centerBox.classList.add("is-dragover");
    });
  });

  ["dragleave", "drop"].forEach((name) => {
    el.centerBox.addEventListener(name, (event) => {
      event.preventDefault();
      event.stopPropagation();
      el.centerBox.classList.remove("is-dragover");
    });
  });

  el.centerBox.addEventListener("drop", (event) => {
    const files = event.dataTransfer && event.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  // Google Drive Popover Toggle & Submit
  el.btnDrive.addEventListener("click", (event) => {
    event.stopPropagation();
    const willOpen = el.drivePopover.hidden;
    closePopovers();
    if (willOpen) {
      el.drivePopover.hidden = false;
      el.btnDrive.setAttribute("aria-expanded", "true");
      el.driveUrlInput.focus();
    }
  });

  el.driveCloseBtn.addEventListener("click", closePopovers);

  el.driveForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const url = el.driveUrlInput.value.trim();
    if (!url) return;
    activeSource = { kind: "drive", url };
    runDriveInference(url);
  });

  // Toggleable Threshold Gate Dropdown Slider
  el.btnThresholdToggle.addEventListener("click", (event) => {
    event.stopPropagation();
    const willOpen = el.thresholdPopover.hidden;
    closePopovers();
    if (willOpen) {
      el.thresholdPopover.hidden = false;
      el.btnThresholdToggle.setAttribute("aria-expanded", "true");
    }
  });

  el.thresholdSlider.addEventListener("input", (event) => {
    const formatted = Number(event.target.value).toFixed(2);
    el.thresholdBtnVal.textContent = formatted;
    el.thresholdPopoverVal.textContent = formatted;
  });

  el.thresholdSlider.addEventListener("change", () => {
    if (!activeSource) return;
    if (activeSource.kind === "file") {
      runFileInference(activeSource.file);
    } else if (activeSource.kind === "drive") {
      runDriveInference(activeSource.url);
    }
  });

  el.newScanBtn.addEventListener("click", returnToLanding);

  // Close popovers on outside click or Escape
  document.addEventListener("click", (event) => {
    const target = event.target;
    if (
      !el.thresholdPopover.contains(target) &&
      !el.btnThresholdToggle.contains(target) &&
      !el.drivePopover.contains(target) &&
      !el.btnDrive.contains(target)
    ) {
      closePopovers();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closePopovers();
    }
  });
})();
