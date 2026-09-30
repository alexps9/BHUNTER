const LEVELS = [
  ["No_Occlusion", "No occlusion"],
  ["Low_Occlusion", "Low"],
  ["Moderate_Occlusion", "Moderate"],
  ["Severe_Occlusion", "Severe"],
  ["Extreme_Occlusion", "Extreme"]
];

const FRAME = {
  No_Occlusion: "000000",
  Low_Occlusion: "000000",
  Moderate_Occlusion: "000004",
  Severe_Occlusion: "000002",
  Extreme_Occlusion: "000000"
};

function detPath(level, phase, sensor) {
  return `media/det/${level}/${phase}/${sensor}.jpg`;
}

function trackPath(level, phase, sensor) {
  return `media/track/${level}/${phase}/${sensor}.mp4`;
}

function press(group, value) {
  group.querySelectorAll("button").forEach((button) => {
    button.setAttribute("aria-pressed", button.dataset.value === value ? "true" : "false");
  });
}

function selected(group) {
  const button = group.querySelector('[aria-pressed="true"]');
  return button ? button.dataset.value : null;
}

function bindGroup(group, onChange) {
  group.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button || !group.contains(button)) return;
    press(group, button.dataset.value);
    onChange(button.dataset.value);
  });
}

function fillLevelTabs(container, onSelect) {
  LEVELS.forEach(([id, label], index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.value = id;
    button.textContent = label;
    button.setAttribute("aria-pressed", index === 0 ? "true" : "false");
    container.appendChild(button);
  });
  bindGroup(container, onSelect);
}

function bindCompare(root) {
  const range = root.querySelector("input");
  const clip = root.querySelector(".clip");
  const handle = root.querySelector(".handle");
  const over = root.querySelector(".over");
  const base = root.querySelector(".base");

  function fitOverlay() {
    over.style.width = `${root.clientWidth}px`;
    over.style.height = `${base.clientHeight}px`;
  }

  function setSplit(value) {
    const pct = `${value}%`;
    clip.style.width = pct;
    handle.style.left = pct;
  }

  range.addEventListener("input", () => setSplit(range.value));
  base.addEventListener("load", fitOverlay);
  window.addEventListener("resize", fitOverlay);
  setSplit(range.value);
  if (base.complete) fitOverlay();
  return fitOverlay;
}

const pairToken = new WeakMap();

function swapPair(beforeImg, afterImg, beforeSrc, afterSrc, fitOverlay) {
  const token = {};
  pairToken.set(beforeImg, token);
  const nextBefore = new Image();
  const nextAfter = new Image();
  let pending = 2;
  const done = () => {
    if (pairToken.get(beforeImg) !== token || --pending !== 0) return;
    beforeImg.src = beforeSrc;
    afterImg.src = afterSrc;
    requestAnimationFrame(fitOverlay);
  };
  nextBefore.onload = nextAfter.onload = done;
  nextBefore.onerror = nextAfter.onerror = done;
  nextBefore.src = beforeSrc;
  nextAfter.src = afterSrc;
}

const galleryCompare = document.getElementById("gallery-compare");
const galleryBefore = document.getElementById("gallery-before");
const galleryAfter = document.getElementById("gallery-after");
const galleryCaption = document.getElementById("gallery-caption");
const galleryThumbs = document.getElementById("gallery-thumbs");
const fitGallery = bindCompare(galleryCompare);

LEVELS.forEach(([id, label], index) => {
  const button = document.createElement("button");
  button.type = "button";
  button.dataset.value = id;
  button.setAttribute("aria-pressed", index === 0 ? "true" : "false");
  button.innerHTML = `<img src="${detPath(id, "after_retrain", "camera")}" alt=""><span>${label}</span>`;
  galleryThumbs.appendChild(button);
});

bindGroup(galleryThumbs, (level) => {
  const label = LEVELS.find(([id]) => id === level)[1];
  galleryBefore.alt = `${label} detections before retraining`;
  galleryAfter.alt = `${label} detections after retraining`;
  galleryCaption.textContent = `${label}. Frame ${FRAME[level]}. Drag to compare before and after retraining.`;
  swapPair(
    galleryBefore,
    galleryAfter,
    detPath(level, "before_retrain", "camera"),
    detPath(level, "after_retrain", "camera"),
    fitGallery
  );
});

const detLevels = document.getElementById("det-levels");
const detSensor = document.getElementById("det-sensor");
const detCaption = document.getElementById("det-caption");
const detCompare = document.getElementById("det-compare");
const detBefore = document.getElementById("det-before");
const detAfter = document.getElementById("det-after");
const fitDetection = bindCompare(detCompare);

function updateDetection() {
  const level = selected(detLevels);
  const sensor = selected(detSensor);
  const label = LEVELS.find(([id]) => id === level)[1];
  const sensorName = sensor === "camera" ? "camera image" : "LiDAR projection";
  detBefore.alt = `${label} detections before retraining on the ${sensorName}`;
  detAfter.alt = `${label} detections after retraining on the ${sensorName}`;
  let note = "";
  if (level === "Severe_Occlusion") {
    note = sensor === "lidar"
      ? " Retraining recovers a second car missed before."
      : " Open LiDAR to see the second car recovered after retraining.";
  }
  detCaption.textContent = `${label}. Blue boxes are model predictions on frame ${FRAME[level]}.${note}`;
  swapPair(
    detBefore,
    detAfter,
    detPath(level, "before_retrain", sensor),
    detPath(level, "after_retrain", sensor),
    fitDetection
  );
}

fillLevelTabs(detLevels, updateDetection);
bindGroup(detSensor, updateDetection);

const trkLevels = document.getElementById("trk-levels");
const trkPhase = document.getElementById("trk-phase");
const trkSensor = document.getElementById("trk-sensor");
const trkVideo = document.getElementById("trk-video");
const trkCaption = document.getElementById("trk-caption");

function updateTracking() {
  const level = selected(trkLevels);
  const phase = selected(trkPhase);
  const sensor = selected(trkSensor);
  const label = LEVELS.find(([id]) => id === level)[1];
  const phaseName = phase === "before_retrain" ? "before retraining" : "after retraining";
  const sensorName = sensor === "camera" ? "camera view" : "LiDAR view";
  const next = trackPath(level, phase, sensor);
  trkVideo.poster = detPath(level, phase, sensor);
  if (trkVideo.getAttribute("src") !== next) {
    trkVideo.src = next;
    trkVideo.load();
  }
  trkCaption.textContent = `${label}, ${phaseName}, ${sensorName}.`;
}

fillLevelTabs(trkLevels, updateTracking);
bindGroup(trkPhase, updateTracking);
bindGroup(trkSensor, updateTracking);

document.getElementById("copy-bib").addEventListener("click", async (event) => {
  const text = document.getElementById("bib-text").textContent;
  const button = event.currentTarget;
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "Copied";
  } catch (error) {
    button.textContent = "Select the text";
  }
  window.setTimeout(() => {
    button.textContent = "Copy";
  }, 1600);
});

document.querySelectorAll(".diagram, .method-steps li, .stats article, .levels article, .reasons li").forEach((el, index) => {
  el.classList.add("reveal");
  el.style.setProperty("--d", `${(index % 5) * 70}ms`);
});

const reveal = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    entry.target.classList.add("in");
    reveal.unobserve(entry.target);
  });
}, { threshold: 0.12 });

document.querySelectorAll(".reveal").forEach((el) => reveal.observe(el));

const sections = [...document.querySelectorAll("main section.page")];
const navLinks = [...document.querySelectorAll(".topnav a")];
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    navLinks.forEach((link) => {
      link.classList.toggle("active", link.getAttribute("href") === `#${entry.target.id}`);
    });
  });
}, { rootMargin: "-40% 0px -50% 0px", threshold: 0.01 });
sections.forEach((section) => observer.observe(section));
