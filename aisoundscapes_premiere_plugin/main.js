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
const { storage } = require("uxp");

const { localFileSystem, formats } = storage;

const API_BASE_URL =
  (typeof window !== "undefined" && window.__AISoundscapesApi?.baseUrl) || "http://localhost:8000";
const PROCESS_FRAMES_ENDPOINT = `${API_BASE_URL.replace(/\/?$/, "")}/process_frames`;
const MAX_FRAMES_PER_REQUEST = 20;

const TICKS_PER_SECOND = 254_016_000_000;

const state = {
  sequenceName: null,
  sequenceIdentifier: null,
  fps: 30,
  salientMoments: [],
  scenes: [],
  pendingSceneStart: null,
  audioResults: [],
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
  generateSoundscape: document.getElementById("generate-soundscape"),
  audioResults: document.getElementById("audio-results"),
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

function slugify(value, fallback = "audio-clip") {
  if (!value || typeof value !== "string") {
    return fallback;
  }
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-{2,}/g, "-")
    .slice(0, 80)
    .trim() || fallback;
}

function base64ToArrayBuffer(base64) {
  const normalized = base64.replace(/\s+/g, "");
  const binaryString = atob(normalized);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i += 1) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}

function resolveNativePath(entry) {
  if (!entry) {
    return null;
  }
  if (entry.nativePath) {
    return entry.nativePath;
  }
  if (entry.url) {
    return entry.url.replace(/^file:\/\//, "");
  }
  return null;
}

function getFrameExportStrategy(sequence) {
  // Debug: Log available methods on the sequence object
  console.log("Sequence object methods:", Object.getOwnPropertyNames(sequence).filter(name => 
    typeof sequence[name] === "function" && name.toLowerCase().includes("export")
  ));
  
  const strategies = [
    {
      method: "exportFrameJPEG",
      extension: "jpg",
      mime: "image/jpeg",
      buildArgs: (ticks, path) => [ticks, path, 1920, 1080, 0.9],
    },
    {
      method: "exportFrameJpeg",
      extension: "jpg",
      mime: "image/jpeg",
      buildArgs: (ticks, path) => [ticks, path, 1920, 1080, 0.9],
    },
    {
      method: "exportFrameJpg",
      extension: "jpg",
      mime: "image/jpeg",
      buildArgs: (ticks, path) => [ticks, path, 1920, 1080, 0.9],
    },
    {
      method: "exportFramePNG",
      extension: "png",
      mime: "image/png",
      buildArgs: (ticks, path) => [ticks, path, 1920, 1080],
    },
    {
      method: "exportFramePng",
      extension: "png",
      mime: "image/png",
      buildArgs: (ticks, path) => [ticks, path, 1920, 1080],
    },
    // Additional fallback strategies
    {
      method: "exportFrame",
      extension: "jpg",
      mime: "image/jpeg",
      buildArgs: (ticks, path) => [ticks, path, 1920, 1080, 0.9],
    },
    {
      method: "exportImage",
      extension: "jpg", 
      mime: "image/jpeg",
      buildArgs: (ticks, path) => [ticks, path, 1920, 1080],
    },
    {
      method: "renderFrame",
      extension: "jpg",
      mime: "image/jpeg", 
      buildArgs: (ticks, path) => [ticks, path],
    },
  ];

  // Try to find a working strategy
  for (const strategy of strategies) {
    if (typeof sequence?.[strategy.method] === "function") {
      console.log(`Found working export method: ${strategy.method}`);
      return strategy;
    }
  }
  
  // If no direct methods work, try accessing through project
  console.log("No direct sequence export methods found. Checking project...");
  return null;
}

async function exportFramesForMoments(sequence, moments) {
  if (!moments?.length) {
    return { frames: [], truncated: false, cleanup: async () => {} };
  }

  const ordered = moments
    .slice()
    .sort((a, b) => a.ticks - b.ticks)
    .slice(0, MAX_FRAMES_PER_REQUEST);

  const truncated = moments.length > ordered.length;
  const strategy = getFrameExportStrategy(sequence);

  if (!strategy) {
    // Try alternative export approach through project
    console.log("Trying project-level export methods...");
    const project = await ppro.Project.getActiveProject();
    
    if (project) {
      // Check for project export methods
      const projectExportMethods = Object.getOwnPropertyNames(project).filter(name => 
        typeof project[name] === "function" && name.toLowerCase().includes("export")
      );
      console.log("Project export methods:", projectExportMethods);
      
      // Try using project export if available
      if (typeof project.exportFrameJPEG === "function") {
        return await exportFramesUsingProject(project, sequence, ordered, truncated);
      }
    }
    
    // Provide detailed error information
    const availableMethods = Object.getOwnPropertyNames(sequence).filter(name => 
      typeof sequence[name] === "function"
    );
    console.log("All sequence methods:", availableMethods);
    
    throw new Error(
      `No frame export method found. Available sequence methods: ${availableMethods.join(", ")}. ` +
      `Please check your Premiere Pro version supports frame export, or try a different approach.`
    );
  }

  const tempFolder = await localFileSystem.createTemporaryFolder();
  const frames = [];

  for (let index = 0; index < ordered.length; index += 1) {
    const moment = ordered[index];
    const safeIndex = pad(index + 1, 2);
    const filename = `salient-${safeIndex}-${moment.frames}.${strategy.extension}`;
    const file = await tempFolder.createFile(filename, { overwrite: true });
    const nativePath = resolveNativePath(file);

    if (!nativePath) {
      throw new Error("Unable to resolve local path for exported frame.");
    }

    const args = strategy.buildArgs(moment.ticks, nativePath);
    const exportOutcome = sequence[strategy.method](...args);
    if (exportOutcome && typeof exportOutcome.then === "function") {
      await exportOutcome;
    }

    const binary = await file.read({ format: formats.binary });
    const blob = new Blob([binary], { type: strategy.mime });

    frames.push({
      moment,
      file,
      filename,
      nativePath,
      blob,
      mime: strategy.mime,
    });
  }

  const cleanup = async () => {
    for (const frame of frames) {
      try {
        if (frame.file?.delete) {
          await frame.file.delete();
        } else if (frame.file?.remove) {
          await frame.file.remove();
        }
      } catch (error) {
        console.warn("Unable to remove exported frame", error);
      }
    }

    try {
      if (tempFolder?.delete) {
        await tempFolder.delete();
      } else if (tempFolder?.remove) {
        await tempFolder.remove();
      }
    } catch (error) {
      console.warn("Unable to remove temporary frame folder", error);
    }
  };

  return { frames, truncated, cleanup };
}

async function exportFramesUsingProject(project, sequence, orderedMoments, truncated) {
  console.log("Using project-level frame export...");
  
  const tempFolder = await localFileSystem.createTemporaryFolder();
  const frames = [];

  for (let index = 0; index < orderedMoments.length; index += 1) {
    const moment = orderedMoments[index];
    const safeIndex = pad(index + 1, 2);
    const filename = `salient-${safeIndex}-${moment.frames}.jpg`;
    const file = await tempFolder.createFile(filename, { overwrite: true });
    const nativePath = resolveNativePath(file);

    if (!nativePath) {
      throw new Error("Unable to resolve local path for exported frame.");
    }

    try {
      // Set playhead to the desired frame first
      try {
        await movePlayheadToTicks(moment.ticks);
      } catch (playheadError) {
        console.warn(`Unable to set playhead to ${moment.timecode}:`, playheadError);
        // Continue with export at current position
      }
      
      // Try different project export signatures
      let exportSuccess = false;
      const exportMethods = [
        () => project.exportFrameJPEG(nativePath, 1920, 1080, 0.9),
        () => project.exportFrame(nativePath, 1920, 1080),
        () => project.exportCurrentFrame(nativePath),
      ];
      
      for (const exportMethod of exportMethods) {
        try {
          if (typeof exportMethod === "function") {
            const result = exportMethod();
            if (result && typeof result.then === "function") {
              await result;
            }
            exportSuccess = true;
            break;
          }
        } catch (methodError) {
          console.warn(`Export method failed:`, methodError);
          continue;
        }
      }
      
      if (!exportSuccess) {
        throw new Error(`Unable to export frame at ${moment.timecode}`);
      }

      const binary = await file.read({ format: formats.binary });
      const blob = new Blob([binary], { type: "image/jpeg" });

      frames.push({
        moment,
        file,
        filename,
        nativePath,
        blob,
        mime: "image/jpeg",
      });
      
    } catch (exportError) {
      console.error(`Failed to export frame ${index + 1}:`, exportError);
      // Continue with other frames even if one fails
    }
  }

  const cleanup = async () => {
    for (const frame of frames) {
      try {
        if (frame.file?.delete) {
          await frame.file.delete();
        } else if (frame.file?.remove) {
          await frame.file.remove();
        }
      } catch (error) {
        console.warn("Unable to remove exported frame", error);
      }
    }

    try {
      if (tempFolder?.delete) {
        await tempFolder.delete();
      } else if (tempFolder?.remove) {
        await tempFolder.remove();
      }
    } catch (error) {
      console.warn("Unable to remove temporary frame folder", error);
    }
  };

  return { frames, truncated, cleanup };
}

async function importAudioIntoProject(nativePath) {
  if (!nativePath) {
    return false;
  }

  const project = await ppro.Project.getActiveProject();
  if (!project || typeof project.importFiles !== "function") {
    return false;
  }

  try {
    let importResult = project.importFiles([nativePath], { suppressUI: true });
    if (importResult && typeof importResult.then === "function") {
      importResult = await importResult;
    }
    if (typeof importResult === "boolean") {
      return importResult;
    }
    if (Array.isArray(importResult)) {
      return importResult.length > 0;
    }
    return true;
  } catch (errorWithOptions) {
    console.warn("Import with options failed, retrying with legacy signature", errorWithOptions);
    try {
      let legacyResult = project.importFiles([nativePath], true);
      if (legacyResult && typeof legacyResult.then === "function") {
        legacyResult = await legacyResult;
      }
      return Boolean(legacyResult);
    } catch (legacyError) {
      console.warn("Unable to import audio file into project", legacyError);
      return false;
    }
  }
}

async function saveAudioClipsToProject(audioClips) {
  if (!audioClips?.length) {
    return [];
  }

  const dataFolder = await localFileSystem.getDataFolder();
  let audioFolder = dataFolder;

  try {
    audioFolder = await dataFolder.createFolder(`ai-soundscapes-audio-${Date.now()}`);
  } catch (creationError) {
    console.warn("Unable to create dedicated audio folder, using data folder root", creationError);
    audioFolder = dataFolder;
  }

  const savedClips = [];

  for (let index = 0; index < audioClips.length; index += 1) {
    const clip = audioClips[index];
    if (!clip?.content_base64) {
      continue;
    }

    const baseName = slugify(clip.fname || clip.filename || clip.label || `clip-${index + 1}`);
    const fileName = baseName.endsWith(".wav") ? baseName : `${baseName}.wav`;

    const file = await audioFolder.createFile(fileName, { overwrite: true });
    const buffer = base64ToArrayBuffer(clip.content_base64);
    await file.write(buffer, { format: formats.binary });

    const nativePath = resolveNativePath(file);

    let imported = false;
    try {
      imported = await importAudioIntoProject(nativePath);
    } catch (importError) {
      console.warn(`Unable to import ${fileName} into project`, importError);
    }

    savedClips.push({
      ...clip,
      fileName,
      filePath: nativePath,
      imported,
      folderPath: resolveNativePath(audioFolder),
    });
  }

  return savedClips;
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
    state.audioResults = [];
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

function renderAudioResults() {
  const container = elements.audioResults;
  if (!container) {
    return;
  }

  container.innerHTML = "";

  if (!state.audioResults.length) {
    container.classList.add("empty");
    const placeholder = document.createElement("sp-body");
    placeholder.classList.add("secondary");
    placeholder.textContent = "No audio clips generated yet.";
    container.appendChild(placeholder);
    return;
  }

  container.classList.remove("empty");

  state.audioResults.forEach((clip, index) => {
    const card = document.createElement("div");
    card.className = "audio-result-card";

    const header = document.createElement("div");
    header.className = "audio-result-card-header";

    const title = document.createElement("strong");
    title.textContent = clip.label || clip.metadata?.title || clip.fname || `Clip ${index + 1}`;

    const status = document.createElement("span");
    status.className = "secondary";
    if (clip.similarity != null && Number.isFinite(clip.similarity)) {
      status.textContent = `Similarity ${(clip.similarity * 100).toFixed(1)}%`;
    } else if (clip.source === "generated") {
      status.textContent = "Generated from prompt";
    } else {
      status.textContent = "Audio suggestion";
    }

    header.append(title, status);
    card.append(header);

    const meta = document.createElement("div");
    meta.className = "audio-result-meta";

    if (clip.filePath) {
      const pathSpan = document.createElement("span");
      pathSpan.textContent = clip.filePath;
      meta.append(pathSpan);
    }

    if (clip.sample_rate) {
      const sampleSpan = document.createElement("span");
      sampleSpan.textContent = `Sample rate ${clip.sample_rate} Hz`;
      meta.append(sampleSpan);
    }

    if (clip.imported) {
      const importedSpan = document.createElement("span");
      importedSpan.textContent = "Imported to project bin";
      meta.append(importedSpan);
    }

    if (clip.metadata?.description) {
      const descriptionSpan = document.createElement("span");
      descriptionSpan.textContent = clip.metadata.description;
      meta.append(descriptionSpan);
    }

    if (meta.childElementCount) {
      card.append(meta);
    }

    container.append(card);
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
  renderAudioResults();
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
  state.audioResults = [];
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

async function generateAudioForKeyframes() {
  if (!state.salientMoments.length) {
    setStatus("Mark at least one salient keyframe before generating audio.", "error");
    return;
  }

  let cleanup = null;

  try {
    const sequence = await ensureSequence();

    setStatus("Exporting salient keyframe frames…");
    const exportResult = await exportFramesForMoments(sequence, state.salientMoments);
    cleanup = exportResult.cleanup;

    if (!exportResult.frames.length) {
      setStatus("Unable to export frames for the selected keyframes.", "error");
      return;
    }

    if (exportResult.truncated) {
      console.warn(`Only the first ${MAX_FRAMES_PER_REQUEST} keyframes were sent to the service.`);
    }

    const formData = new FormData();
    exportResult.frames.forEach((frame) => {
      formData.append("frames", frame.blob, frame.filename);
    });

    setStatus("Uploading frames to audio service…");

    const response = await fetch(PROCESS_FRAMES_ENDPOINT, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => "");
      throw new Error(
        `Audio service error (${response.status}): ${errorText || response.statusText || "Unknown error"}`,
      );
    }

    const payload = await response.json();

    const audioPayloads = [];
    if (payload?.generated_audio?.content_base64) {
      audioPayloads.push({
        ...payload.generated_audio,
        label: "Generated soundscape",
        source: "generated",
        metadata: {
          description: payload.audio_description || "",
          ...(payload.generated_audio.metadata || {}),
        },
      });
    }

    if (Array.isArray(payload?.similar_clips)) {
      payload.similar_clips.forEach((clip, index) => {
        if (clip?.content_base64) {
          audioPayloads.push({
            ...clip,
            label: clip.metadata?.title || clip.fname || `Similar clip ${index + 1}`,
            source: "library",
          });
        }
      });
    }

    if (!audioPayloads.length) {
      state.audioResults = [];
      renderAudioResults();
      setStatus("Audio service returned no playable clips.", "error");
      return;
    }

    setStatus("Saving audio clips to project…");
    const savedClips = await saveAudioClipsToProject(audioPayloads);
    state.audioResults = savedClips;
    renderAudioResults();

    const importedCount = savedClips.filter((clip) => clip.imported).length;
    const savedCount = savedClips.length;
    let successMessage = `Saved ${savedCount} audio clip${savedCount === 1 ? "" : "s"}.`;
    if (importedCount) {
      successMessage = `Imported ${importedCount} audio clip${importedCount === 1 ? "" : "s"} into the project.`;
    }

    if (exportResult.truncated) {
      successMessage += ` (Only the first ${MAX_FRAMES_PER_REQUEST} keyframes were processed.)`;
    }

    setStatus(successMessage, "success");
  } catch (error) {
    console.error("generateAudioForKeyframes error", error);
    setStatus(error.message || "Unable to generate audio from keyframes.", "error");
  } finally {
    if (cleanup) {
      try {
        await cleanup();
      } catch (cleanupError) {
        console.warn("Unable to clean up exported frames", cleanupError);
      }
    }
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
  elements.generateSoundscape?.addEventListener("click", generateAudioForKeyframes);
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
