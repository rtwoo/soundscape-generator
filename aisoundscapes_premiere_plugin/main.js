/*************************************************************************
 * ADOBE CONFIDENTIAL
 * ___________________
 *
 * Copyright 2024 Adobe
 * All Rights Reserved.
 *
 * NOTICE: Adobe permits you to use, modify, and distribute this file in
 * accordance with the terms of the Adobe license agreement accompanying
 * it. If you have received this file from a source other than Adobe,
 * then your use, modification, or distribution of it requires the prior
 * written permission of Adobe.
 **************************************************************************/

const ppro = require("premierepro");

const TICKS_PER_SECOND = 254_016_000_000;

const state = {
  sequenceName: null,
  sequenceIdentifier: null,
  fps: 30,
  salientMoments: [],
  scenes: [],
  pendingSceneStart: null,
};

const elements = {
  sequenceName: document.getElementById("sequence-name"),
  sequenceFps: document.getElementById("sequence-fps"),
  salientList: document.getElementById("salient-list"),
  sceneList: document.getElementById("scene-list"),
  pendingScene: document.getElementById("pending-scene"),
  status: document.getElementById("status-message"),
  markSalient: document.getElementById("mark-salient"),
  markSceneStart: document.getElementById("mark-scene-start"),
  markSceneEnd: document.getElementById("mark-scene-end"),
  refreshSequence: document.getElementById("refresh-sequence"),
  copyAnnotations: document.getElementById("copy-annotations"),
  clearAnnotations: document.getElementById("clear-annotations"),
};

function createId(prefix) {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
}

function pad(value, size = 2) {
  return value.toString().padStart(size, "0");
}

function formatTimecodeFromSeconds(seconds, fps) {
  const totalFrames = Math.round(seconds * fps);
  const framesPerHour = fps * 3600;
  const framesPerMinute = fps * 60;

  const hours = Math.floor(totalFrames / framesPerHour);
  const minutes = Math.floor((totalFrames % framesPerHour) / framesPerMinute);
  const secs = Math.floor((totalFrames % framesPerMinute) / fps);
  const frames = totalFrames % fps;

  return `${pad(hours)}:${pad(minutes)}:${pad(secs)}:${pad(frames)}`;
}

function buildTimestamp(ticks, fps) {
  const numericTicks = Number(ticks);
  const seconds = numericTicks / TICKS_PER_SECOND;
  const frames = Math.round(seconds * fps);

  return {
    ticks: numericTicks,
    seconds,
    frames,
    timecode: formatTimecodeFromSeconds(seconds, fps),
  };
}

function setStatus(message = "", tone = "neutral") {
  if (!elements.status) {
    return;
  }

  elements.status.textContent = message;
  elements.status.classList.remove("status-success", "status-error", "secondary");

  if (!message) {
    elements.status.classList.add("secondary");
    return;
  }

  if (tone === "success") {
    elements.status.classList.add("status-success");
  } else if (tone === "error") {
    elements.status.classList.add("status-error");
  } else {
    elements.status.classList.add("secondary");
  }
}

function resetSequenceDisplay() {
  if (elements.sequenceName) {
    elements.sequenceName.textContent = "No active sequence";
  }
  if (elements.sequenceFps) {
    elements.sequenceFps.textContent = "Frame rate: -- fps";
  }
}

function normaliseRate(rate) {
  if (!rate) {
    return null;
  }
  if (typeof rate === "number") {
    return rate;
  }
  if (typeof rate === "string") {
    const parsed = Number(rate);
    return Number.isFinite(parsed) ? parsed : null;
  }
  if (typeof rate === "object") {
    const { numerator, denominator, fps, value } = rate;
    if (Number.isFinite(numerator) && Number.isFinite(denominator) && denominator !== 0) {
      return numerator / denominator;
    }
    if (Number.isFinite(rate.fps)) {
      return rate.fps;
    }
    if (Number.isFinite(rate.value)) {
      return rate.value;
    }
    if (fps) {
      return normaliseRate(fps);
    }
    if (value) {
      return normaliseRate(value);
    }
  }
  return null;
}

async function resolveSequenceFps(sequence) {
  let fps = state.fps || 30;

  try {
    if (typeof sequence.getSettings === "function") {
      const settings = await sequence.getSettings();
      const candidates = [
        settings?.videoFrameRate,
        settings?.frameRate,
        settings?.timebase,
        settings?.video?.frameRate,
      ];

      for (const candidate of candidates) {
        const maybeFps = normaliseRate(candidate);
        if (maybeFps) {
          fps = maybeFps;
          break;
        }
      }

      if (!fps && Number.isFinite(settings?.fps)) {
        fps = settings.fps;
      }
    }
  } catch (error) {
    console.warn("Unable to read sequence settings", error);
  }

  if (!fps || !Number.isFinite(fps)) {
    fps = 30;
  }

  return fps;
}

async function syncSequenceMetadata(sequence) {
  const identifier =
    sequence?.nodeId || sequence?.sequenceID || sequence?.guid || sequence?.id || sequence?.name;
  let cleared = false;

  if (state.sequenceIdentifier && identifier && state.sequenceIdentifier !== identifier) {
    state.salientMoments = [];
    state.scenes = [];
    state.pendingSceneStart = null;
    cleared = true;
  }

  state.sequenceIdentifier = identifier || state.sequenceIdentifier;
  const sequenceName = sequence?.name || "Untitled sequence";
  state.sequenceName = sequenceName;

  const fps = await resolveSequenceFps(sequence);
  state.fps = fps;

  if (elements.sequenceName) {
    elements.sequenceName.textContent = sequenceName;
  }
  if (elements.sequenceFps) {
    const displayFps = fps % 1 === 0 ? fps.toFixed(0) : fps.toFixed(3);
    elements.sequenceFps.textContent = `Frame rate: ${displayFps} fps`;
  }

  return cleared;
}

async function ensureSequence() {
  const project = await ppro.Project.getActiveProject();
  if (!project) {
    throw new Error("There is no active project.");
  }

  const sequence = await project.getActiveSequence();
  if (!sequence) {
    throw new Error("There is no active sequence.");
  }

  const cleared = await syncSequenceMetadata(sequence);
  if (cleared) {
    render();
    setStatus("Sequence changed – annotations reset for safety.");
  }

  return sequence;
}

async function getCurrentTimestamp() {
  const sequence = await ensureSequence();
  const position = await sequence.getPlayerPosition();
  let ticks = null;

  if (typeof position === "number") {
    ticks = position;
  } else if (position && typeof position === "object") {
    if (Number.isFinite(position.ticks)) {
      ticks = position.ticks;
    } else if (Number.isFinite(position.seconds)) {
      ticks = position.seconds * TICKS_PER_SECOND;
    }
  }

  if (!Number.isFinite(ticks)) {
    throw new Error("Unable to determine the current playhead position.");
  }

  return buildTimestamp(ticks, state.fps);
}

async function movePlayheadToTicks(ticks) {
  const numericTicks = Number(ticks);
  if (!Number.isFinite(numericTicks)) {
    throw new Error("Invalid timestamp for playhead movement.");
  }

  const sequence = await ensureSequence();
  const strategies = [
    { method: "setPlayerPosition", args: [numericTicks] },
    { method: "setPlayerPosition", args: [{ ticks: numericTicks }] },
    { method: "setPlayerPositionInTicks", args: [numericTicks] },
    { method: "setPlayerPositionTicks", args: [numericTicks] },
  ];

  let lastError = null;

  for (const strategy of strategies) {
    const candidate = sequence[strategy.method];
    if (typeof candidate !== "function") {
      continue;
    }

    try {
      const result = candidate.apply(sequence, strategy.args);
      if (result && typeof result.then === "function") {
        await result;
      }
      return;
    } catch (error) {
      lastError = error;
    }
  }

  if (sequence.playhead && typeof sequence.playhead.setPosition === "function") {
    try {
      const outcome = sequence.playhead.setPosition(numericTicks);
      if (outcome && typeof outcome.then === "function") {
        await outcome;
      }
      return;
    } catch (error) {
      lastError = error;
    }
  }

  if (lastError) {
    throw lastError;
  }

  throw new Error("Unable to move the sequence playhead in Premiere Pro.");
}

async function movePlayheadToMoment(moment) {
  if (!moment || !Number.isFinite(moment.ticks)) {
    throw new Error("Keyframe timestamp is unavailable.");
  }

  await movePlayheadToTicks(moment.ticks);
}

function renderSalientList() {
  const container = elements.salientList;
  if (!container) {
    return;
  }

  container.innerHTML = "";

  if (!state.salientMoments.length) {
    container.classList.add("empty");
    const emptyElement = document.createElement("sp-body");
    emptyElement.classList.add("secondary");
    emptyElement.textContent = "No keyframes marked yet.";
    container.appendChild(emptyElement);
    return;
  }

  container.classList.remove("empty");

  state.salientMoments
    .slice()
    .sort((a, b) => a.ticks - b.ticks)
    .forEach((moment, index) => {
  const row = document.createElement("div");
  row.classList.add("annotation-row", "annotation-row--interactive");
      row.dataset.id = moment.id;

      const meta = document.createElement("div");
      meta.className = "annotation-meta";

      const label = document.createElement("span");
      label.className = "label";
      label.textContent = `Moment ${index + 1}`;

      const timecode = document.createElement("span");
      timecode.className = "timecode";
      timecode.textContent = moment.timecode;

      const metaDetail = document.createElement("span");
      metaDetail.className = "secondary";
      metaDetail.textContent = `Frame ${moment.frames} • ${moment.seconds.toFixed(3)}s`;

      meta.append(label, timecode, metaDetail);

      const sceneField = document.createElement("div");
      sceneField.className = "annotation-scene-field";

      const sceneFieldLabel = document.createElement("span");
      sceneFieldLabel.className = "scene-field-label";
      sceneFieldLabel.textContent = "Scenes";

      const sceneFieldValue = document.createElement("div");
      sceneFieldValue.className = "scene-field-value";
      if (Array.isArray(moment.associatedScenes) && moment.associatedScenes.length) {
        moment.associatedScenes.forEach((association) => {
          const pill = document.createElement("span");
          pill.className = "pill";
          pill.textContent = `Scene ${association.sceneOrder}`;
          sceneFieldValue.appendChild(pill);
        });
      } else {
        const noneLabel = document.createElement("span");
        noneLabel.classList.add("secondary");
        noneLabel.textContent = "None";
        sceneFieldValue.appendChild(noneLabel);
      }

      sceneField.append(sceneFieldLabel, sceneFieldValue);

      const actions = document.createElement("div");
      actions.className = "annotation-actions";

      const removeButton = document.createElement("sp-button");
      removeButton.setAttribute("variant", "secondary");
      removeButton.setAttribute("size", "s");
      removeButton.dataset.action = "remove-salient";
      removeButton.dataset.id = moment.id;
      removeButton.textContent = "Remove";

  actions.appendChild(removeButton);
  row.append(meta, sceneField, actions);
      container.appendChild(row);
    });
}

function recomputeSceneAssociations() {
  if (!state.scenes.length) {
    state.salientMoments.forEach((moment) => {
      moment.sceneId = null;
      moment.sceneOrder = null;
      moment.associatedScenes = [];
    });
    return;
  }

  const sortedScenes = state.scenes.slice().sort((a, b) => a.start.ticks - b.start.ticks);
  const sortedMoments = state.salientMoments.slice().sort((a, b) => a.ticks - b.ticks);
  const momentAssignments = new Map();

  sortedScenes.forEach((scene, index) => {
    const order = index + 1;
    scene.order = order;

    const associated = sortedMoments.filter(
      (moment) => moment.ticks >= scene.start.ticks && moment.ticks <= scene.end.ticks,
    );

    scene.salientMomentIds = associated.map((moment) => moment.id);
    scene.salientMoments = associated.map((moment) => ({
      id: moment.id,
      timecode: moment.timecode,
      seconds: moment.seconds,
      frames: moment.frames,
      ticks: moment.ticks,
    }));

    associated.forEach((moment) => {
      const existing = momentAssignments.get(moment.id) || [];
      existing.push({ sceneId: scene.id, sceneOrder: order });
      momentAssignments.set(moment.id, existing);
    });
  });

  state.salientMoments.forEach((moment) => {
    const assignments = momentAssignments.get(moment.id) || [];
    moment.associatedScenes = assignments;
    if (assignments.length) {
      moment.sceneId = assignments[0].sceneId;
      moment.sceneOrder = assignments[0].sceneOrder;
    } else {
      moment.sceneId = null;
      moment.sceneOrder = null;
    }
  });
}

function renderSceneList() {
  const container = elements.sceneList;
  if (!container) {
    return;
  }

  recomputeSceneAssociations();
  renderSalientList();

  container.innerHTML = "";

  if (!state.scenes.length) {
    container.classList.add("empty");
    const emptyElement = document.createElement("sp-body");
    emptyElement.classList.add("secondary");
    emptyElement.textContent = "No scenes defined yet.";
    container.appendChild(emptyElement);
    return;
  }

  container.classList.remove("empty");

  state.scenes
    .slice()
    .sort((a, b) => a.start.ticks - b.start.ticks)
    .forEach((scene, index) => {
      const row = document.createElement("div");
      row.className = "annotation-row";
      row.dataset.id = scene.id;

      const meta = document.createElement("div");
      meta.className = "annotation-meta";

      const label = document.createElement("span");
      label.className = "label";
      const sceneOrder = Number.isFinite(scene.order) ? scene.order : index + 1;
      label.textContent = `Scene ${sceneOrder}`;

      const timecode = document.createElement("span");
      timecode.className = "timecode";
      timecode.textContent = `${scene.start.timecode} → ${scene.end.timecode}`;

      const metaDetail = document.createElement("span");
      metaDetail.className = "secondary";
      metaDetail.textContent = `${scene.duration.timecode} • ${scene.duration.seconds.toFixed(3)}s`;

      const associatedMoments = scene.salientMoments || [];
      const keyframeDetail = document.createElement("span");
      keyframeDetail.className = "secondary";
      if (associatedMoments.length) {
        const momentTimecodes = associatedMoments.map((moment) => moment.timecode).join(", ");
        keyframeDetail.textContent = `Keyframes: ${momentTimecodes}`;
      } else {
        keyframeDetail.textContent = "Keyframes: none";
      }

      meta.append(label, timecode, metaDetail, keyframeDetail);

      const actions = document.createElement("div");
      actions.className = "annotation-actions";

      const removeButton = document.createElement("sp-button");
      removeButton.setAttribute("variant", "secondary");
      removeButton.setAttribute("size", "s");
      removeButton.dataset.action = "remove-scene";
      removeButton.dataset.id = scene.id;
      removeButton.textContent = "Remove";

      actions.appendChild(removeButton);
      row.append(meta, actions);
      container.appendChild(row);
    });
}

function renderPendingScene() {
  if (!elements.pendingScene) {
    return;
  }

  if (state.pendingSceneStart) {
    elements.pendingScene.textContent = `Pending start: ${state.pendingSceneStart.timecode}`;
    elements.pendingScene.classList.remove("secondary");
  } else {
    elements.pendingScene.textContent = "";
    elements.pendingScene.classList.add("secondary");
  }
}

function render() {
  renderSceneList();
  renderPendingScene();
}

async function handleSalientListClick(event) {
  const actionEl = event.target.closest("[data-action]");
  if (actionEl && actionEl.dataset.action === "remove-salient") {
    const id = actionEl.dataset.id;
    state.salientMoments = state.salientMoments.filter((moment) => moment.id !== id);
    renderSceneList();
    setStatus("Removed keyframe annotation.");
    return;
  }

  const row = event.target.closest(".annotation-row");
  if (!row || !row.dataset.id) {
    return;
  }

  const moment = state.salientMoments.find((candidate) => candidate.id === row.dataset.id);
  if (!moment) {
    setStatus("Unable to find that keyframe.", "error");
    return;
  }

  try {
    await movePlayheadToMoment(moment);
    setStatus(`Moved playhead to ${moment.timecode}.`, "success");
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Unable to move the playhead.", "error");
  }
}

function handleSceneListClick(event) {
  const actionEl = event.target.closest("[data-action]");
  if (!actionEl) {
    return;
  }

  if (actionEl.dataset.action === "remove-scene") {
    const id = actionEl.dataset.id;
    state.scenes = state.scenes.filter((scene) => scene.id !== id);
    renderSceneList();
    setStatus("Removed scene annotation.");
  }
}

async function addSalientMoment() {
  try {
    const timestamp = await getCurrentTimestamp();
    const id = createId("salient");
    state.salientMoments.push({ id, ...timestamp });
    state.salientMoments.sort((a, b) => a.ticks - b.ticks);
    renderSceneList();
    setStatus(`Marked salient moment at ${timestamp.timecode}.`, "success");
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Unable to mark salient moment.", "error");
  }
}

async function markSceneStart() {
  try {
    const timestamp = await getCurrentTimestamp();
    state.pendingSceneStart = timestamp;
    renderPendingScene();
    setStatus(`Scene start set to ${timestamp.timecode}. Set an end to save.`, "success");
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Unable to set scene start.", "error");
  }
}

async function markSceneEnd() {
  if (!state.pendingSceneStart) {
    setStatus("Set a scene start before marking an end.", "error");
    return;
  }

  try {
    const endTimestamp = await getCurrentTimestamp();
    const start = state.pendingSceneStart;

    if (endTimestamp.ticks <= start.ticks) {
      setStatus("Scene end must be after the start position.", "error");
      return;
    }

    const durationSeconds = endTimestamp.seconds - start.seconds;
    const durationTimecode = formatTimecodeFromSeconds(durationSeconds, state.fps);
    const durationFrames = Math.round(durationSeconds * state.fps);

    const newStartTicks = start.ticks;
    const newEndTicks = endTimestamp.ticks;

    const overlappingScene = state.scenes.find((existing) => {
      if (!existing?.start || !existing?.end) {
        return false;
      }
      const existingStart = Number(existing.start.ticks);
      const existingEnd = Number(existing.end.ticks);
      if (!Number.isFinite(existingStart) || !Number.isFinite(existingEnd)) {
        return false;
      }
      return newStartTicks < existingEnd && newEndTicks > existingStart;
    });

    if (overlappingScene) {
      setStatus("Scene overlaps with an existing scene. Adjust the start or end point.", "error");
      return;
    }

    const associatedMoments = state.salientMoments
      .filter((moment) => Number.isFinite(moment.ticks) && moment.ticks >= newStartTicks && moment.ticks <= newEndTicks)
      .map((moment) => ({
        id: moment.id,
        timecode: moment.timecode,
        seconds: moment.seconds,
        frames: moment.frames,
        ticks: moment.ticks,
      }));

    const scene = {
      id: createId("scene"),
      start,
      end: endTimestamp,
      duration: {
        seconds: durationSeconds,
        frames: durationFrames,
        timecode: durationTimecode,
      },
      salientMomentIds: associatedMoments.map((moment) => moment.id),
      salientMoments: associatedMoments,
    };

    state.scenes.push(scene);
    state.scenes.sort((a, b) => a.start.ticks - b.start.ticks);
    state.pendingSceneStart = null;
    renderSceneList();
    renderPendingScene();
    setStatus(
      `Scene captured (${scene.start.timecode} → ${scene.end.timecode}, ${scene.duration.timecode}).`,
      "success",
    );
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Unable to set scene end.", "error");
  }
}

async function refreshSequence() {
  try {
    const sequence = await ensureSequence();
    setStatus(`Connected to "${sequence.name}".`, "success");
    render();
  } catch (error) {
    console.error(error);
    resetSequenceDisplay();
    setStatus(error.message || "Unable to access the active sequence.", "error");
  }
}

function clearAnnotations() {
  if (!state.salientMoments.length && !state.scenes.length && !state.pendingSceneStart) {
    setStatus("Nothing to clear.");
    return;
  }

  state.salientMoments = [];
  state.scenes = [];
  state.pendingSceneStart = null;
  render();
  setStatus("Cleared all annotations.");
}

async function copyAnnotationsToClipboard() {
  const payload = {
    generatedAt: new Date().toISOString(),
    sequence: {
      name: state.sequenceName,
      fps: state.fps,
    },
    salientMoments: state.salientMoments.map((moment, index) => ({
      id: moment.id,
      order: index + 1,
      timecode: moment.timecode,
      seconds: moment.seconds,
      frames: moment.frames,
      ticks: moment.ticks,
      sceneId: moment.sceneId || null,
      sceneOrder: Number.isFinite(moment.sceneOrder) ? moment.sceneOrder : null,
      associatedScenes: Array.isArray(moment.associatedScenes)
        ? moment.associatedScenes.map((association) => ({
            sceneId: association.sceneId,
            sceneOrder: association.sceneOrder,
          }))
        : [],
    })),
    scenes: state.scenes.map((scene, index) => ({
      id: scene.id,
      order: Number.isFinite(scene.order) ? scene.order : index + 1,
      start: scene.start,
      end: scene.end,
      duration: scene.duration,
      salientMomentIds: scene.salientMomentIds || [],
      salientMoments: (scene.salientMoments || []).map((moment) => ({
        id: moment.id,
        timecode: moment.timecode,
        seconds: moment.seconds,
        frames: moment.frames,
        ticks: moment.ticks,
      })),
      salientMomentCount: Array.isArray(scene.salientMomentIds)
        ? scene.salientMomentIds.length
        : 0,
    })),
  };

  const json = JSON.stringify(payload, null, 2);

  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(json);
    } else if (require("uxp").clipboard) {
      require("uxp").clipboard.copyText(json);
    } else {
      throw new Error("Clipboard API is unavailable.");
    }
    setStatus("Copied annotations to clipboard.", "success");
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Unable to copy annotations.", "error");
  }
}

function registerEventListeners() {
  elements.salientList?.addEventListener("click", handleSalientListClick);
  elements.sceneList?.addEventListener("click", handleSceneListClick);
  elements.markSalient?.addEventListener("click", addSalientMoment);
  elements.markSceneStart?.addEventListener("click", markSceneStart);
  elements.markSceneEnd?.addEventListener("click", markSceneEnd);
  elements.refreshSequence?.addEventListener("click", refreshSequence);
  elements.clearAnnotations?.addEventListener("click", clearAnnotations);
  elements.copyAnnotations?.addEventListener("click", copyAnnotationsToClipboard);
}

function init() {
  registerEventListeners();
  render();
  refreshSequence();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
