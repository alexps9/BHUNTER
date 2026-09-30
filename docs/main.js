const LEVELS = [
  ["No_Occlusion", "No occlusion"],
  ["Low_Occlusion", "Low"],
  ["Moderate_Occlusion", "Moderate"],
  ["Severe_Occlusion", "Severe"],
  ["Extreme_Occlusion", "Extreme"]
];

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

const galleryImage = document.getElementById("gallery-image");
const galleryCaption = document.getElementById("gallery-caption");
const galleryThumbs = document.getElementById("gallery-thumbs");

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
  galleryImage.src = detPath(level, "after_retrain", "camera");
  galleryImage.alt = `Predicted detection under ${label.toLowerCase()}`;
  galleryCaption.textContent = `${label}. Frame 000000, after retraining.`;
});

const detLevels = document.getElementById("det-levels");
const detSensor = document.getElementById("det-sensor");
const detCaption = document.getElementById("det-caption");
const detBefore = document.getElementById("det-before");
const detAfter = document.getElementById("det-after");

function updateDetection() {
  const level = selected(detLevels);
  const sensor = selected(detSensor);
  const label = LEVELS.find(([id]) => id === level)[1];
  const sensorName = sensor === "camera" ? "camera image" : "LiDAR projection";
  detBefore.src = detPath(level, "before_retrain", sensor);
  detAfter.src = detPath(level, "after_retrain", sensor);
  detBefore.alt = `${label} detections before retraining on the ${sensorName}`;
  detAfter.alt = `${label} detections after retraining on the ${sensorName}`;
  detCaption.textContent = `${label}. Blue boxes are model predictions on frame 000000.`;
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

const sections = [...document.querySelectorAll("main section")];
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
