# AI Soundscape Generator Panel

This Premiere Pro UXP panel lets editors capture sequence context for an AI-driven soundscape generator. Use it to mark salient keyframes that should feed object/salience detection and to define scene ranges that determine how much audio needs to be synthesized.

## Features

- **Sequence awareness** – inspect and refresh the currently active sequence name and frame rate.
- **Salient keyframes** – capture the playhead position as a keyframe marker, ordered by time.
- **Scene ranges** – set a start and end to describe clip-length segments and auto-calculate their duration.
- **Clipboard export** – copy the annotations as structured JSON for downstream services.

## Getting Started

1. Load the plugin with UXP Developer Tools. Add the folder containing `manifest.json` to your workspace and click **Load**.
2. Dock or float the panel in Premiere Pro. The header updates to the active sequence when detected.
3. Navigate your timeline and use **Mark current playhead** to add salient keyframes.
4. For scenes, click **Set start** at the beginning and **Set end** at the end of the region. A scene entry is added automatically.
5. Use **Copy JSON** to place the annotations on the clipboard. The payload includes timestamps in timecode, seconds, frames, and ticks.
6. Click **Clear** to reset the annotations (handy when switching sequences).

Keep the panel open while editing. It will reset annotations automatically if it detects that you switched to a new sequence to prevent cross-sequence data leakage.
