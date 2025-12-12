# Realtime-AV-Swap

**Realtime-AV-Swap** is an integrated system that combines real-time voice conversion and real-time face swapping into a synchronized audio-visual stream.

It orchestrates two independent subsystems:
1.  **Audio**: [Seed-VC Realtime](https://github.com/jiaheguo521/seed-vc-realtime) for low-latency voice conversion.
2.  **Video**: [Deep-Live-Cam](https://github.com/jiaheguo521/Deep-Live-Cam) for real-time face swapping.

A custom **Shared Memory** mechanism is used to synchronize audio and video latencies, ensuring lip-sync is maintained even when processing delays fluctuate.

## Features

*   **Dual-Process Architecture**: Audio and Video run in separate processes.
*   **Unified Launcher**: A single script (`launcher.py`) manages the startup and lifecycle of both subsystems.
*   **Unified GUI Launcher**: `unified_launcher.py` provides a single GUI for controlling both subsystems in one process.

## Prerequisites

*   **OS**: Windows 10/11 (The launcher is currently configured for Windows paths).
*   **Hardware**: NVIDIA GPU with CUDA support is highly recommended for real-time performance.
*   **Python**: Python 3.10.

## Installation

### Common Setup
1.  **Clone this repository**:
    ```bash
    git clone https://github.com/YourUsername/Realtime-AV-Swap.git
    cd Realtime-AV-Swap
    ```

2.  **Create and Activate Virtual Environment**:
    It is recommended to create a virtual environment to isolate dependencies.
    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate the virtual environment (Windows)
    .\venv\Scripts\activate
    ```

### Option 1: Unified Launcher (GUI)
If you want to use `unified_launcher.py`, follow these steps:

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ```
2.  **Note**: This method runs both Audio and Video modules in the same environment. Ensure you have the submodules (`audio/seed-vc-realtime` and `video/Deep-Live-Cam`) cloned, but you do not necessarily need their individual virtual environments if you install all dependencies in the root environment.

### Option 2: Dual-Process Launcher (Legacy)

2.  **Setup Audio Subsystem**:
    *   Navigate to the `audio` directory.
    *   Clone the `seed-vc-realtime` repository.
    *   Create a virtual environment named `venv` inside `audio/seed-vc-realtime`.
    *   Install dependencies as per the submodule's instructions.
    ```bash
    # Example structure expectation:
    # audio/seed-vc-realtime/venv/Scripts/python.exe
    ```

3.  **Setup Video Subsystem**:
    *   Navigate to the `video` directory.
    *   Clone the `Deep-Live-Cam` repository.
    *   Create a virtual environment named `venv` inside `video/Deep-Live-Cam`.
    *   Install dependencies (ensure CUDA providers are installed).
    ```bash
    # Example structure expectation:
    # video/Deep-Live-Cam/venv/Scripts/python.exe
    ```

## Project Structure

```
Realtime-AV-Swap/
├── audio/
│   └── seed-vc-realtime/    # Voice Conversion Subsystem
├── video/
│   └── Deep-Live-Cam/       # Face Swap Subsystem
├── launcher.py              # Main entry point
├── DELAY_LOGIC.md           # Documentation on sync logic
└── README.md
```

## Usage

### Unified Launcher (GUI)
Run the unified GUI launcher:
```bash
python unified_launcher.py
```

### Dual-Process Launcher
1.  Ensure both virtual environments are set up correctly.
2.  Run the launcher from the root directory:

```bash
python launcher.py
```

The launcher will:
1.  Initialize the Shared Memory block.
2.  Start the Audio GUI.
3.  Start the Video Live Cam (with CUDA execution provider).
4.  Monitor both processes.

### Controls
*   **Stop**: Press `Ctrl+C` in the terminal running `launcher.py` to gracefully shut down both processes and clean up shared memory.

## Synchronization Logic

The system uses a "wait-for-slowest" strategy:
*   **Audio Process** reports its physical input/output latency + buffer time.
*   **Video Process** reports its processing time per frame.
*   If Video is slower, Audio waits (buffers output).
*   If Audio is slower, Video holds the frame.


