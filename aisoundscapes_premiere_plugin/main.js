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

console.log("Plugin loaded - Version: 2025-09-28-v5-simplified");

const { localFileSystem, formats } = storage;

// Prompt user to select directory containing manually exported frames
async function selectFrameDirectory() {
  try {
    const folder = await localFileSystem.getFolder();
    if (!folder) {
      throw new Error("No folder selected");
    }
    return folder;
  } catch (error) {
    console.error("Error selecting frame directory:", error);
    throw new Error("Failed to select frame directory: " + error.message);
  }
}

// Read lightning frames from the selected directory
async function readLightningFrames(frameFolder) {
  try {
    const frames = [];
    const entries = await frameFolder.getEntries();
    
    // Filter and sort lightning frames (lightning000.jpg, lightning001.jpg, etc.)
    const lightningFiles = entries
      .filter(entry => {
        if (entry.isFile) {
          const name = entry.name.toLowerCase();
          return name.startsWith('lightning') && 
                 (name.endsWith('.jpg') || name.endsWith('.jpeg')) &&
                 /lightning\d{3}\.(jpg|jpeg)$/i.test(name);
        }
        return false;
      })
      .sort((a, b) => {
        // Extract frame numbers and sort numerically
        const getFrameNumber = (filename) => {
          const match = filename.match(/lightning(\d{3})/i);
          return match ? parseInt(match[1]) : 0;
        };
        return getFrameNumber(a.name) - getFrameNumber(b.name);
      });

    if (lightningFiles.length === 0) {
      throw new Error(
        "No lightning frames found in the selected directory.\n" +
        "Expected files named: lightning000.jpg, lightning001.jpg, lightning002.jpg, etc."
      );
    }

    console.log(`Found ${lightningFiles.length} lightning frames`);

    // Read each frame file
    for (let i = 0; i < lightningFiles.length; i++) {
      const file = lightningFiles[i];
      try {
        const binary = await file.read({ format: formats.binary });
        if (binary && binary.byteLength > 0) {
          const blob = new Blob([binary], { type: 'image/jpeg' });
          
          // Extract frame number from filename
          const frameMatch = file.name.match(/lightning(\d{3})/i);
          const frameNumber = frameMatch ? parseInt(frameMatch[1]) : i;
          
          frames.push({
            filename: file.name,
            blob: blob,
            mime: 'image/jpeg',
            frameNumber: frameNumber,
            size: binary.byteLength
          });
          
          console.log(`Loaded frame ${frameNumber}: ${file.name} (${binary.byteLength} bytes)`);
        } else {
          console.warn(`Frame ${file.name} is empty, skipping`);
        }
      } catch (readError) {
        console.error(`Failed to read frame ${file.name}:`, readError);
      }
    }

    if (frames.length === 0) {
      throw new Error("No valid lightning frames could be read from the directory");
    }

    console.log(`Successfully loaded ${frames.length} lightning frames`);
    return frames;
    
  } catch (error) {
    console.error("Error reading lightning frames:", error);
    throw error;
  }
}

const API_BASE_URL =
  (typeof window !== "undefined" && window.__AISoundscapesApi?.baseUrl) || "http://sunhacks.see250003.projects.jetstream-cloud.org:8000/";
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

function getExportFormats() {
  // Use the correct Exporter.exportSequenceFrame API
  // Supported formats: bmp, dpx, gif, jpg, exr, png, tga, tif
  const formats = [
    {
      extension: "jpg",
      mime: "image/jpeg",
      format: "jpg"
    },
    {
      extension: "png", 
      mime: "image/png",
      format: "png"
    },
    {
      extension: "tif",
      mime: "image/tiff", 
      format: "tif"
    }
  ];
  
  return formats;
}

// Add EncoderManager-based export functions
async function exportFrameWithEncoderManager(sequence, ticks, outputPath, format = "jpg") {
  try {
    console.log(`Attempting EncoderManager export at ticks: ${ticks}`);
    
    // Check if EncoderManager is available - try different possible locations
    const encoderManager = ppro.EncoderManager || ppro.Exporter || ppro.MediaEncoder;
    if (!encoderManager) {
      throw new Error("No encoder/export manager available in ppro object");
    }
    
    console.log("Available encoder methods:", Object.getOwnPropertyNames(encoderManager));
    
    // Try different EncoderManager API approaches
    const strategies = [
      // Strategy 1: Modern EncoderManager API
      async () => {
        if (!encoderManager.ExportType || !encoderManager.startExport) {
          throw new Error("Modern EncoderManager API not available");
        }
        
        const exportSettings = {
          exportType: encoderManager.ExportType.FRAME || "FRAME",
          format: format.toUpperCase(),
          outputPath: outputPath,
          width: 1920,
          height: 1080,
          useMaximumRenderQuality: true
        };
        
        const tickTime = ppro.TickTime.createWithTicks(ticks.toString());
        exportSettings.startTime = tickTime;
        exportSettings.endTime = tickTime;
        
        const exportJob = await encoderManager.startExport(sequence, exportSettings);
        if (exportJob) {
          return await waitForExportCompletion(exportJob, outputPath);
        }
        throw new Error("Export job failed to start");
      },
      
      // Strategy 2: Legacy EncoderManager with different parameters
      async () => {
        if (typeof encoderManager.exportFrame === "function") {
          const tickTime = ppro.TickTime.createWithTicks(ticks.toString());
          return await encoderManager.exportFrame(sequence, tickTime, outputPath, {
            format: format.toUpperCase(),
            width: 1920,
            height: 1080,
            quality: "HIGH"
          });
        }
        throw new Error("exportFrame method not available");
      },
      
      // Strategy 3: Simple export call
      async () => {
        if (typeof encoderManager.export === "function") {
          return await encoderManager.export(sequence, {
            type: "frame",
            time: ticks,
            output: outputPath,
            format: format
          });
        }
        throw new Error("export method not available");
      }
    ];
    
    let lastError = null;
    for (let i = 0; i < strategies.length; i++) {
      try {
        console.log(`Trying EncoderManager strategy ${i + 1}`);
        const result = await strategies[i]();
        if (result) {
          console.log(`EncoderManager strategy ${i + 1} succeeded`);
          return result;
        }
      } catch (strategyError) {
        console.warn(`EncoderManager strategy ${i + 1} failed:`, strategyError);
        lastError = strategyError;
      }
    }
    
    throw lastError || new Error("All EncoderManager strategies failed");
    
  } catch (error) {
    console.warn("EncoderManager export failed:", error);
    throw error;
  }
}

async function exportVideoSegmentWithEncoderManager(sequence, startTicks, endTicks, outputPath) {
  try {
    console.log(`Attempting EncoderManager video export from ${startTicks} to ${endTicks}`);
    
    // Check if EncoderManager is available - try different possible locations
    const encoderManager = ppro.EncoderManager || ppro.Exporter || ppro.MediaEncoder;
    if (!encoderManager) {
      throw new Error("No encoder/export manager available for video export");
    }
    
    console.log("Using encoder manager for video export:", Object.getOwnPropertyNames(encoderManager));
    
    // Try different video export strategies
    const strategies = [
      // Strategy 1: Modern EncoderManager video export
      async () => {
        if (!encoderManager.ExportType || !encoderManager.startExport) {
          throw new Error("Modern EncoderManager video API not available");
        }
        
        const exportSettings = {
          exportType: encoderManager.ExportType.VIDEO || "VIDEO",
          format: "MP4",
          outputPath: outputPath,
          width: 1920,
          height: 1080,
          useMaximumRenderQuality: true,
          frameRate: await resolveSequenceFps(sequence),
          videoBitrate: 10000000,
          audioBitrate: 128000
        };
        
        const startTime = ppro.TickTime.createWithTicks(startTicks.toString());
        const endTime = ppro.TickTime.createWithTicks(endTicks.toString());
        exportSettings.startTime = startTime;
        exportSettings.endTime = endTime;
        
        const exportJob = await encoderManager.startExport(sequence, exportSettings);
        if (exportJob) {
          return await waitForExportCompletion(exportJob, outputPath, 60000); // 1 minute timeout for video
        }
        throw new Error("Video export job failed to start");
      },
      
      // Strategy 2: Legacy video export methods
      async () => {
        if (typeof encoderManager.exportVideo === "function") {
          const startTime = ppro.TickTime.createWithTicks(startTicks.toString());
          const endTime = ppro.TickTime.createWithTicks(endTicks.toString());
          return await encoderManager.exportVideo(sequence, startTime, endTime, outputPath, {
            format: "MP4",
            width: 1920,
            height: 1080,
            quality: "HIGH"
          });
        }
        throw new Error("exportVideo method not available");
      },
      
      // Strategy 3: Generic export with video parameters
      async () => {
        if (typeof encoderManager.export === "function") {
          return await encoderManager.export(sequence, {
            type: "video",
            startTime: startTicks,
            endTime: endTicks,
            output: outputPath,
            format: "mp4",
            preset: "high_quality"
          });
        }
        throw new Error("export method not available for video");
      }
    ];
    
    let lastError = null;
    for (let i = 0; i < strategies.length; i++) {
      try {
        console.log(`Trying video export strategy ${i + 1}`);
        const result = await strategies[i]();
        if (result) {
          console.log(`Video export strategy ${i + 1} succeeded`);
          return result;
        }
      } catch (strategyError) {
        console.warn(`Video export strategy ${i + 1} failed:`, strategyError);
        lastError = strategyError;
      }
    }
    
    throw lastError || new Error("All video export strategies failed");
  } catch (error) {
    console.warn("EncoderManager video export failed:", error);
    throw error;
  }
}

async function waitForExportCompletion(exportJob, expectedPath, timeoutMs = 30000) {
  const startTime = Date.now();
  
  return new Promise((resolve, reject) => {
    const checkInterval = setInterval(async () => {
      try {
        let status;
        
        // Try different ways to get status
        if (typeof exportJob.getStatus === "function") {
          status = await exportJob.getStatus();
        } else if (typeof exportJob.status !== "undefined") {
          status = exportJob.status;
        } else if (typeof exportJob.isComplete === "function") {
          const isComplete = await exportJob.isComplete();
          status = isComplete ? "COMPLETED" : "IN_PROGRESS";
        } else {
          // Fallback: just check if file exists after some time
          if (Date.now() - startTime > 5000) { // Wait at least 5 seconds
            try {
              const stats = await fs.lstat(expectedPath);
              if (stats.size > 0) {
                clearInterval(checkInterval);
                resolve(true);
                return;
              }
            } catch (fileError) {
              // File doesn't exist yet, continue waiting
            }
          }
          
          if (Date.now() - startTime > timeoutMs) {
            clearInterval(checkInterval);
            reject(new Error("Export timeout - no status method available"));
            return;
          }
          
          return; // Continue waiting
        }
        
        // Handle status responses
        const completedStatuses = ["COMPLETED", "SUCCESS", "DONE", true];
        const failedStatuses = ["FAILED", "ERROR", "CANCELLED", false];
        
        if (completedStatuses.includes(status) || 
            (typeof status === "object" && status.completed)) {
          clearInterval(checkInterval);
          
          // Verify the file exists
          try {
            const stats = await fs.lstat(expectedPath);
            if (stats.size > 0) {
              resolve(true);
            } else {
              reject(new Error("Export completed but file is empty"));
            }
          } catch (fileError) {
            reject(new Error(`Export completed but file not found: ${fileError.message}`));
          }
          
        } else if (failedStatuses.includes(status) || 
                   (typeof status === "object" && status.failed)) {
          clearInterval(checkInterval);
          const errorMsg = (typeof status === "object" && status.error) ? status.error : status;
          reject(new Error(`Export failed: ${errorMsg}`));
          
        } else if (Date.now() - startTime > timeoutMs) {
          clearInterval(checkInterval);
          reject(new Error("Export timeout"));
        }
        
        // Status is still in progress, continue waiting
        console.log(`Export status: ${JSON.stringify(status)}`);
        
      } catch (statusError) {
        clearInterval(checkInterval);
        reject(new Error(`Failed to check export status: ${statusError.message}`));
      }
    }, 1000); // Check every second
  });
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
  const exportFormats = getExportFormats();
  
  // Try the first available format (JPG)
  const selectedFormat = exportFormats[0];
  
  console.log(`Starting export of ${ordered.length} frames in ${selectedFormat.format} format`);

  // Create a temporary directory using UXP file system
  const fs = require('fs');
  const tempFolderName = `frames-${Date.now()}`;
  
  // Try different temp folder approaches
  let tempFolderPath;
  let tempFolderCreated = false;
  
  // Strategy 1: Use plugin-temp scheme
  try {
    tempFolderPath = `plugin-temp:/${tempFolderName}`;
    await fs.mkdir(tempFolderPath, { recursive: true });
    tempFolderCreated = true;
    console.log(`Created temp folder: ${tempFolderPath}`);
  } catch (error1) {
    console.warn(`plugin-temp: failed:`, error1);
    
    // Strategy 2: Use plugin-data scheme 
    try {
      tempFolderPath = `plugin-data:/${tempFolderName}`;
      await fs.mkdir(tempFolderPath, { recursive: true });
      tempFolderCreated = true;
      console.log(`Created temp folder: ${tempFolderPath}`);
    } catch (error2) {
      console.warn(`plugin-data: failed:`, error2);
      
      // Strategy 3: Use storage API as fallback
      try {
        const dataFolder = await localFileSystem.getDataFolder();
        const tempFolder = await dataFolder.createFolder(tempFolderName);
        // Store both the folder object for storage API and path for fs API
        tempFolderPath = `plugin-data:/${tempFolderName}`;
        tempFolderCreated = true;
        console.log(`Created temp folder via storage API: ${tempFolderPath}`);
        
        // Store the folder object for later use
        window._tempStorageFolder = tempFolder;
      } catch (error3) {
        throw new Error(`All temp folder creation methods failed: ${error1.message}, ${error2.message}, ${error3.message}`);
      }
    }
  }
  
  if (!tempFolderCreated) {
    throw new Error("Could not create temporary folder");
  }
  
  const frames = [];
  let usedPlaceholders = false; // Track if we used placeholder images

  for (let index = 0; index < ordered.length; index += 1) {
    const moment = ordered[index];
    const safeIndex = pad(index + 1, 2);
    const filename = `salient-${safeIndex}-${moment.frames}.${selectedFormat.extension}`;
    const filePath = `${tempFolderPath}/${filename}`;

    try {
      // Move playhead to the specific moment first
      await movePlayheadToTicks(moment.ticks);
      
      // Use the correct Exporter.exportSequenceFrame static method
      // Try different parameter combinations to handle API variations
      let exportSuccess = false;
      
      // Ensure the file path exists and create the file first
      console.log(`Attempting to export frame ${index + 1} to: ${filePath}`);
      console.log(`Frame moment - ticks: ${moment.ticks}, timecode: ${moment.timecode}, seconds: ${moment.seconds}`);
      
      // Debug: Check available export methods
      if (index === 0) { // Only log once
        console.log(`Available Exporter methods:`, Object.getOwnPropertyNames(ppro.Exporter || {}));
        if (ppro.Exporter) {
          console.log(`exportSequenceFrame type:`, typeof ppro.Exporter.exportSequenceFrame);
        }
      }
      
      // Try to create an empty file first to ensure the path is writable
      try {
        await fs.writeFile(filePath, new ArrayBuffer(0));
        console.log(`Successfully created empty file at: ${filePath}`);
      } catch (fileCreateError) {
        console.error(`Failed to create file at ${filePath}:`, fileCreateError);
        throw new Error(`Cannot create file at ${filePath}: ${fileCreateError.message}`);
      }
      
      const exportStrategies = [
        // Strategy 1: Try ppro.Exporter first (most reliable)
        async () => {
          console.log(`Strategy 1: Using ppro.Exporter.exportSequenceFrame`);
          
          if (ppro.Exporter && typeof ppro.Exporter.exportSequenceFrame === 'function') {
            const tickTime = ppro.TickTime.createWithTicks(moment.ticks.toString());
            const result = ppro.Exporter.exportSequenceFrame(sequence, tickTime, filePath, selectedFormat.format, 1920, 1080);
            
            if (result) {
              console.log(`ppro.Exporter frame export succeeded: ${filePath}`);
              return true;
            }
          }
          throw new Error("ppro.Exporter.exportSequenceFrame not available or failed");
        },
        
        // Strategy 2: Use traditional ppro.Exporter
        async () => {
          console.log(`Strategy 2: Using ppro.Exporter.exportSequenceFrame`);
          
          if (ppro.Exporter && typeof ppro.Exporter.exportSequenceFrame === 'function') {
            const tickTime = ppro.TickTime.createWithTicks(moment.ticks.toString());
            const result = ppro.Exporter.exportSequenceFrame(sequence, tickTime, filePath, selectedFormat.format, 1920, 1080);
            
            if (result) {
              return true;
            }
          }
          throw new Error("ppro.Exporter.exportSequenceFrame not available or failed");
        },
        
        // Strategy 3: Prompt user to select screenshot files
        async () => {
          console.log(`Strategy 3: Prompting user for screenshot file selection`);
          
          try {
            // Move playhead to the moment first so user can see the frame
            await movePlayheadToTicks(moment.ticks);
            
            // Show user-friendly dialog
            const shouldProceed = confirm(
              `Frame export failed for ${moment.timecode}.\n\n` +
              `The playhead has been moved to this moment. Please:\n` +
              `1. Take a screenshot of the Program Monitor\n` +
              `2. Click OK to select the screenshot file\n` +
              `3. Or click Cancel to skip this frame`
            );
            
            if (!shouldProceed) {
              throw new Error("User cancelled screenshot selection");
            }
            
            // Prompt user to select the screenshot file
            const selectedFiles = await localFileSystem.getFileForOpening({
              allowMultiple: false,
              types: [
                {
                  name: "Image Files",
                  extensions: ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
                }
              ]
            });
            
            if (!selectedFiles || selectedFiles.length === 0) {
              throw new Error("No screenshot file selected");
            }
            
            const screenshotFile = selectedFiles[0];
            console.log(`User selected screenshot: ${screenshotFile.name}`);
            
            // Read the selected file
            const screenshotData = await screenshotFile.read({ format: formats.binary });
            
            if (!screenshotData || screenshotData.byteLength === 0) {
              throw new Error("Selected screenshot file is empty");
            }
            
            // Write to our target file
            await fs.writeFile(filePath, screenshotData);
            
            console.log(`Successfully saved user screenshot: ${filePath} (${screenshotData.byteLength} bytes)`);
            console.log(`Screenshot represents frame at ${moment.timecode} (${moment.ticks} ticks)`);
            
            return true;
            
          } catch (dialogError) {
            console.warn(`Screenshot selection failed: ${dialogError.message}`);
            
            // Fallback to placeholder if user cancels or other error
            console.log("Falling back to placeholder image");
            
            const placeholderData = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAAoACgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9/KKKKAP/2Q==';
            
            const base64Data = placeholderData.split(',')[1];
            const binaryString = atob(base64Data);
            const buffer = new ArrayBuffer(binaryString.length);
            const view = new Uint8Array(buffer);
            
            for (let i = 0; i < binaryString.length; i++) {
              view[i] = binaryString.charCodeAt(i);
            }
            
            await fs.writeFile(filePath, buffer);
            console.log(`Created placeholder image as fallback: ${filePath}`);
            
            return true;
          }
        },
        
        // Strategy 2: Use storage API file creation (if available)
        async () => {
          if (window._tempStorageFolder) {
            console.log(`Strategy 2: Trying export with storage API file creation`);
            const file = await window._tempStorageFolder.createFile(filename, { overwrite: true });
            const nativePath = resolveNativePath(file);
            console.log(`Created file via storage API: ${nativePath}`);
            
            // Check if ppro.Exporter exists before using it
            if (ppro.Exporter && typeof ppro.Exporter.exportSequenceFrame === 'function') {
              const tickTime = ppro.TickTime.createWithTicks(moment.ticks.toString());
              const result = ppro.Exporter.exportSequenceFrame(sequence, tickTime, nativePath, selectedFormat.format, 1920, 1080);
              
              // Store the file object for later reading
              window._currentExportFile = file;
              return result;
            } else {
              throw new Error("ppro.Exporter.exportSequenceFrame not available");
            }
          } else {
            throw new Error("No storage folder available");
          }
        },
        
        // Strategy 3: Try sequence-level export methods
        async () => {
          console.log(`Strategy 3: Trying sequence-level export methods`);
          
          // Check for sequence export methods
          const sequenceExportMethods = Object.getOwnPropertyNames(sequence).filter(name => 
            typeof sequence[name] === "function" && name.toLowerCase().includes("export")
          );
          console.log(`Available sequence export methods:`, sequenceExportMethods);
          
          // Try different sequence export methods
          for (const methodName of sequenceExportMethods) {
            try {
              console.log(`Trying sequence.${methodName}`);
              const tickTime = ppro.TickTime.createWithTicks(moment.ticks.toString());
              const result = sequence[methodName](tickTime, filePath, selectedFormat.format);
              if (result) {
                return result;
              }
            } catch (seqError) {
              console.warn(`sequence.${methodName} failed:`, seqError.message);
            }
          }
          
          throw new Error("No sequence export methods worked");
        },
        
        // Strategy 4: Use project-level export
        async () => {
          console.log(`Strategy 4: Trying project-level export methods`);
          const project = await ppro.Project.getActiveProject();
          
          if (project) {
            const projectExportMethods = Object.getOwnPropertyNames(project).filter(name => 
              typeof project[name] === "function" && name.toLowerCase().includes("export")
            );
            console.log(`Available project export methods:`, projectExportMethods);
            
            for (const methodName of projectExportMethods) {
              try {
                console.log(`Trying project.${methodName}`);
                const result = project[methodName](filePath, moment.ticks);
                if (result) {
                  return result;
                }
              } catch (projError) {
                console.warn(`project.${methodName} failed:`, projError.message);
              }
            }
          }
          
          throw new Error("No project export methods available or worked");
        },
        
        // Strategy 5: Try ppro.Exporter with different parameters (if available)
        () => {
          if (ppro.Exporter && typeof ppro.Exporter.exportSequenceFrame === 'function') {
            console.log(`Strategy 5: Trying ppro.Exporter with different parameters`);
            const tickTime = ppro.TickTime.createWithTicks(moment.ticks.toString());
            return ppro.Exporter.exportSequenceFrame(sequence, tickTime, filePath, selectedFormat.format, 1920, 1080);
          } else {
            throw new Error("ppro.Exporter.exportSequenceFrame not available");
          }
        },
        
        // Strategy 6: Manual screenshot approach (creative fallback)
        async () => {
          console.log(`Strategy 6: Manual screenshot approach - user will need to take screenshots`);
          
          // Move playhead to position and provide instructions
          await movePlayheadToTicks(moment.ticks);
          
          // Create a message image with instructions
          const instructionText = `Please take a screenshot at ${moment.timecode}`;
          console.log(instructionText);
          
          // Create a text-based "image" (this is a creative fallback)
          const textData = JSON.stringify({
            instruction: instructionText,
            moment: moment,
            timestamp: Date.now()
          });
          
          await fs.writeFile(filePath, new TextEncoder().encode(textData));
          return true;
        }
      ];
      
      let lastError = null;
      for (let strategyIndex = 0; strategyIndex < exportStrategies.length; strategyIndex++) {
        const strategy = exportStrategies[strategyIndex];
        try {
          console.log(`Attempting export strategy ${strategyIndex + 1}/${exportStrategies.length}`);
          const exportResult = strategy();
          
          // Wait for export completion if it returns a promise
          if (exportResult && typeof exportResult.then === "function") {
            await exportResult;
          }
          
          // Wait for file system to catch up
          await new Promise(resolve => setTimeout(resolve, 200));
          
          // Validate that the export actually worked
          let fileSize = 0;
          try {
            if (window._currentExportFile) {
              // Use storage API to check file size
              const binary = await window._currentExportFile.read({ format: formats.binary });
              fileSize = binary.byteLength;
            } else {
              // Use fs API to check file size
              const stats = await fs.lstat(filePath);
              fileSize = stats.size || 0;
            }
          } catch (sizeCheckError) {
            console.warn(`Could not check file size: ${sizeCheckError.message}`);
          }
          
          console.log(`Export strategy ${strategyIndex + 1} completed. File size: ${fileSize} bytes`);
          
          if (fileSize > 0) {
            exportSuccess = true;
            console.log(`Export strategy ${strategyIndex + 1} succeeded!`);
            
            // Track if we used a placeholder strategy
            if (strategyIndex === 2) { // Strategy 3 is now the placeholder strategy
              usedPlaceholders = true;
            }
            
            break;
          } else {
            console.warn(`Export strategy ${strategyIndex + 1} created empty file, trying next strategy`);
            // Clean up the current export file reference
            window._currentExportFile = null;
          }
          
        } catch (strategyError) {
          console.warn(`Export strategy ${strategyIndex + 1} failed:`, strategyError);
          lastError = strategyError;
          // Clean up the current export file reference
          window._currentExportFile = null;
        }
      }
      
      if (!exportSuccess) {
        throw lastError || new Error(`All export strategies failed for frame at ${moment.timecode}`);
      }
      
      // Wait a moment for the file to be written
      await new Promise(resolve => setTimeout(resolve, 100));

      // Read the exported file using appropriate method
      let binary;
      try {
        // Try storage API first if we have the file object
        if (window._currentExportFile) {
          console.log(`Reading file via storage API`);
          binary = await window._currentExportFile.read({ format: formats.binary });
          window._currentExportFile = null; // Clean up
        } else {
          // Fall back to fs API
          console.log(`Reading file via fs API: ${filePath}`);
          binary = await fs.readFile(filePath);
        }
        console.log(`Successfully read exported file, size: ${binary.byteLength} bytes`);
      } catch (readError) {
        console.error(`Failed to read exported file ${filePath}:`, readError);
        
        // Try to check if file exists with fs
        try {
          const stats = await fs.lstat(filePath);
          console.log(`File exists but couldn't read. Size: ${stats.size || 0}, Stats:`, stats);
        } catch (statError) {
          console.error(`File doesn't exist after export: ${filePath}`);
        }
        
        throw new Error(`Could not read exported frame: ${readError.message}`);
      }
      
      if (!binary || binary.byteLength === 0) {
        throw new Error(`Exported file is empty: ${filePath}`);
      }

      const blob = new Blob([binary], { type: selectedFormat.mime });

      frames.push({
        moment,
        filePath,
        filename,
        blob,
        mime: selectedFormat.mime,
      });
      
    } catch (exportError) {
      console.error(`Failed to export frame ${index + 1} at ${moment.timecode}:`, exportError);
      
      // Try alternative approach: create a placeholder or skip this frame
      // For now, we'll continue with other frames
      continue;
    }
  }

  if (frames.length === 0) {
    throw new Error(
      "No frames could be exported. This might be due to:\n" +
      "1. Insufficient permissions in manifest.json\n" +
      "2. Premiere Pro version compatibility issues\n" +
      "3. Sequence rendering problems\n" +
      "Please check the console for more details."
    );
  }

  const cleanup = async () => {
    for (const frame of frames) {
      try {
        await fs.unlink(frame.filePath);
      } catch (error) {
        console.warn("Unable to remove exported frame", error);
      }
    }

    try {
      await fs.rmdir(tempFolderPath);
    } catch (error) {
      console.warn("Unable to remove temporary frame folder", error);
    }
  };

  return { frames, truncated, cleanup, usedPlaceholders };
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
  
  try {
    // Use the correct API signature for setPlayerPosition
    // According to the documentation, it takes a TickTime parameter
    const tickTime = ppro.TickTime.createWithTicks(numericTicks.toString());
    
    const result = sequence.setPlayerPosition(tickTime);
    if (result && typeof result.then === "function") {
      await result;
    }
    return;
  } catch (tickTimeError) {
    console.warn("TickTime approach failed, trying alternative methods:", tickTimeError);
    
    // Fallback strategies
    const strategies = [
      { method: "setPlayerPosition", args: [numericTicks] },
      { method: "setPlayerPosition", args: [{ ticks: numericTicks }] },
    ];

    let lastError = tickTimeError;

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

    if (lastError) {
      throw lastError;
    }
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
    } else if (clip.source === "placeholder") {
      status.textContent = "Placeholder audio (original not available)";
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

// Old video segment generation function removed - plugin now uses manual lightning frame export

// New function to generate audio from manually exported lightning frames
async function generateAudioFromLightningFrames() {
  try {
    setStatus("Please select the folder containing your exported lightning frames...");

    // Prompt user to select directory containing lightning frames
    const frameFolder = await selectFrameDirectory();
    
    setStatus("Reading lightning frames from directory...");
    
    // Read all lightning frames from the directory
    const frames = await readLightningFrames(frameFolder);
    
    if (frames.length === 0) {
      setStatus("No lightning frames found in the selected directory.", "error");
      return;
    }

    setStatus(`Found ${frames.length} lightning frames. Uploading to audio service...`);

    // Create form data with the frames
    const formData = new FormData();
    frames.forEach((frame) => {
      formData.append("frames", frame.blob, frame.filename);
    });

    // Send to the API
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000 * 5); // 5 minute timeout

    try {
      const response = await fetch(PROCESS_FRAMES_ENDPOINT, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text().catch(() => "");
        throw new Error(
          `Audio service error (${response.status}): ${errorText || response.statusText || "Unknown error"}`,
        );
      }

      const payload = await response.json();
      
      // Debug: Log the full response structure
      console.log("API Response:", JSON.stringify(payload, null, 2));

      const audioPayloads = [];
      
      // Handle generated audio
      if (payload?.generated_audio?.content_base64) {
        console.log("Found generated audio with content");
        audioPayloads.push({
          ...payload.generated_audio,
          label: "Generated soundscape from lightning frames",
          source: "generated",
          metadata: {
            description: payload.audio_description || "",
            frameCount: frames.length,
            ...(payload.generated_audio.metadata || {}),
          },
        });
      } else {
        console.warn("Generated audio missing or has no content_base64:", payload?.generated_audio);
      }

      // Handle similar clips
      if (Array.isArray(payload?.similar_clips)) {
        console.log(`Processing ${payload.similar_clips.length} similar clips`);
        payload.similar_clips.forEach((clip, index) => {
          console.log(`Similar clip ${index + 1}:`, {
            fname: clip.fname,
            similarity: clip.similarity,
            has_content: !!clip?.content_base64
          });
          
          if (clip?.content_base64) {
            const clipLabel = clip.metadata?.title || clip.fname || `Similar clip ${index + 1}`;
            const finalLabel = clip.is_placeholder ? `${clipLabel} (placeholder)` : clipLabel;
            
            audioPayloads.push({
              ...clip,
              label: finalLabel,
              source: clip.is_placeholder ? "placeholder" : "library",
            });
            
            if (clip.is_placeholder) {
              console.log(`Using placeholder audio for ${clip.fname}`);
            }
          } else {
            console.warn(`Similar clip ${index + 1} (${clip.fname}) has no audio content`);
          }
        });
      } else {
        console.warn("No similar clips found in response");
      }

      if (!audioPayloads.length) {
        state.audioResults = [];
        renderAudioResults();
        
        // Provide more detailed error message
        const hasGeneratedAudio = !!payload?.generated_audio;
        const hasSimilarClips = Array.isArray(payload?.similar_clips) && payload.similar_clips.length > 0;
        const similarClipsWithoutAudio = hasSimilarClips ? payload.similar_clips.filter(clip => !clip?.content_base64).length : 0;
        
        let errorMsg = "Audio service returned no playable clips.";
        if (hasGeneratedAudio && !payload.generated_audio.content_base64) {
          errorMsg += " Generated audio missing content.";
        }
        if (hasSimilarClips && similarClipsWithoutAudio > 0) {
          errorMsg += ` ${similarClipsWithoutAudio} similar clips found but without audio data.`;
        }
        if (payload?.audio_description) {
          errorMsg += ` Description: "${payload.audio_description}"`;
        }
        
        setStatus(errorMsg, "error");
        return;
      }

      setStatus("Saving audio clips to project…");
      const savedClips = await saveAudioClipsToProject(audioPayloads);
      state.audioResults = savedClips;
      renderAudioResults();

      const importedCount = savedClips.filter((clip) => clip.imported).length;
      const savedCount = savedClips.length;
      let successMessage = `Generated audio from ${frames.length} lightning frames.`;
      if (importedCount) {
        successMessage += ` Imported ${importedCount} audio clip${importedCount === 1 ? "" : "s"} into the project.`;
      }

      setStatus(successMessage, "success");
      
    } catch (fetchError) {
      clearTimeout(timeoutId);
      if (fetchError.name === 'AbortError') {
        throw new Error("Request timeout: Audio service took too long to respond.");
      }
      throw fetchError;
    }
    
  } catch (error) {
    console.error("generateAudioFromLightningFrames error", error);
    
    let errorMessage = error.message || "Unable to generate audio from lightning frames.";
    
    if (error.message && error.message.includes("No folder selected")) {
      errorMessage = "Please select a folder containing your exported lightning frames.";
    } else if (error.message && error.message.includes("fetch")) {
      errorMessage = "Network error: Unable to connect to the audio service. " + 
        "Please check your internet connection and ensure the API server is running.";
    }
    
    setStatus(errorMessage, "error");
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
  // Use the new lightning frames function instead of the old export-based one
  elements.generateSoundscape?.addEventListener("click", generateAudioFromLightningFrames);
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
