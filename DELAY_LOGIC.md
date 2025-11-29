# Realtime-AV-Swap Synchronization Logic

This document details the cross-process synchronization mechanism used to align audio (Seed-VC) and video (Deep-Live-Cam) streams in the Realtime-AV-Swap project.

## 1. Architecture Overview

The system consists of two independent processes managed by a `launcher.py`. Synchronization is achieved via **Shared Memory** (IPC).

- **Architecture**: Peer-to-Peer Synchronization.
- **Principle**: Each process reports its own internal latency and reads the other process's latency as a "Target Delay".
- **Goal**: The faster process waits for the slower process, ensuring alignment at the output.

## 2. Shared Memory Layout

*   **Name**: `RAS_SharedMem`
*   **Size**: 24 bytes
*   **Format**: 3 x `double` (8 bytes each)

| Offset | Type | Description | Writer | Reader |
| :--- | :--- | :--- | :--- | :--- |
| `0:8` | `double` | **Audio Delay (ms)** | Audio Process | Video Process |
| `8:16` | `double` | **Video Delay (ms)** | Video Process | Audio Process |
| `16:24` | `double` | Target Delay (Legacy) | Launcher | Unused |

## 3. Audio Logic (`seed-vc-realtime`)

### A. Delay Calculation
The audio process calculates its "Physical Delay" — the time it takes for audio to travel from microphone input to speaker output, plus algorithmic latency.

**Formula**:
```python
self.delay_time = (
    input_latency       # Hardware Input Latency
    + output_latency    # Hardware Output Latency
    + block_time        # Audio Block Duration (Buffer size)
    + crossfade_time    # Overlap-Add Crossfade duration
    + 0.01              # Safety Buffer
    + 0.20              # MANUAL OFFSET (200ms) - Fine-tuning to match Video
)
```
*Note: The Manual Offset (0.20s) is hardcoded to compensate for unmeasured system latencies (e.g., Windows mixer, driver buffers).*

### B. Reporting
Every audio callback cycle:
1. Converts `self.delay_time` to milliseconds.
2. Writes to Shared Memory `[0:8]`.

### C. Synchronization (Waiting)
1. Reads **Video Delay** from Shared Memory `[8:16]`.
2. Calculates `extra_delay`:
   `extra = max(0, Video_Delay - Audio_Delay)`
3. **Buffering**: Pushes processed audio into a `delay_queue`.
4. **Release**: Pops audio from queue only when:
   `Current_Time - Chunk_Timestamp >= extra_delay`

### D. Shutdown
On `stop_stream`, writes `0.0` to Shared Memory `[0:8]` to release the video process from waiting.

## 4. Video Logic (`Deep-Live-Cam`)

### A. Delay Calculation
The video process measures processing time per frame.
`current_delay_ms = (Current_Time - Capture_Time) * 1000`

### B. Reporting
Writes `current_delay_ms` to Shared Memory `[8:16]`.

### C. Synchronization (Frame Buffering)
The video pipeline does **not** sleep (blocking execution would lower FPS). Instead, it uses a `deque` (Frame Buffer).

1. **Capture**: Frame is captured, processed, and pushed to `frame_buffer` with a timestamp.
2. **Read Target**: Reads **Audio Delay** from Shared Memory `[0:8]`.
3. **Safety Clamps**:
   - `target = min(target, 1.0)`: Never wait more than 1 second.
   - `target == 0.0`: **Real-time Mode**. If audio reports 0 (inactive), sync is disabled, and video runs FIFO.

### D. Display Strategy
Iterates through `frame_buffer` to find the best frame to display:

*   **Condition**: `(Current_Time - Frame_Timestamp) >= Target_Audio_Delay`
*   **Skip/Drop Logic**:
    - If multiple frames meet the condition (video is lagging behind audio or audio delay dropped), it **skips** older frames and picks the *newest* ready frame.
    - This prevents the "fast-forward" effect and ensures lipsync catches up quickly.
*   **Hold Logic**:
    - If *no* frames meet the condition (video is ahead of audio), it displays nothing new (holds the previous frame) until the oldest frame in the buffer "ages" enough.

## 5. Summary of Data Flow

1. **Audio** starts -> Calculates ~500ms physical delay -> Writes to SHM.
2. **Video** starts -> Reads 500ms target -> buffers frames until they are 500ms old -> Displays.
3. **Video** processing spikes -> Reports higher delay (e.g., 600ms) to SHM.
4. **Audio** reads 600ms target -> Sees it is faster (500ms < 600ms) -> Holds audio output for extra 100ms.
5. **Audio** stops -> Writes 0.0ms to SHM.
6. **Video** reads 0.0ms -> Disables buffer wait -> Displays frames immediately.

