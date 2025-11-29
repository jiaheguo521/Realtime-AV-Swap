import multiprocessing.shared_memory
import subprocess
import time
import os
import struct
import sys

# Constants
SHM_SIZE = 24  # 3 floats (8 bytes each)
# Offset 0: Audio Delay (ms)
# Offset 8: Video Delay (ms)
# Offset 16: Target Delay (ms)

def main():
    print("Initializing Realtime-AV-Swap Launcher...")
    
    # Create Shared Memory
    shm_name = 'RAS_SharedMem'
    try:
        # Try to create
        shm = multiprocessing.shared_memory.SharedMemory(name=shm_name, create=True, size=SHM_SIZE)
    except FileExistsError:
        # If exists, try to connect and close/unlink, then recreate
        try:
            shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
        except:
            pass # It might have been cleaned up or I don't have permission
        
        try:
            shm = multiprocessing.shared_memory.SharedMemory(name=shm_name, create=True, size=SHM_SIZE)
        except Exception as e:
            print(f"Failed to create shared memory with name {shm_name}: {e}")
            # Fallback to anonymous
            shm = multiprocessing.shared_memory.SharedMemory(create=True, size=SHM_SIZE)
            shm_name = shm.name
            
    print(f"Shared Memory Created: {shm_name}")
    
    # Initialize to 0.0
    shm.buf[:SHM_SIZE] = bytearray([0] * SHM_SIZE)

    # Paths
    root_dir = os.getcwd()
    
    audio_dir = os.path.join(root_dir, "audio", "seed-vc-realtime")
    audio_python = os.path.join(audio_dir, "venv", "Scripts", "python.exe")
    audio_script = os.path.join(audio_dir, "real-time-gui.py")
    
    video_dir = os.path.join(root_dir, "video", "Deep-Live-Cam")
    video_python = os.path.join(video_dir, "venv", "Scripts", "python.exe")
    video_script = os.path.join(video_dir, "run.py")

    if not os.path.exists(audio_python):
        print(f"Error: Audio python not found at {audio_python}")
        return
    if not os.path.exists(video_python):
        print(f"Error: Video python not found at {video_python}")
        return

    # Environment variables
    env = os.environ.copy()
    env["RAS_SHARED_MEM_NAME"] = shm_name

    processes = []
    
    try:
        print("Starting Audio Project (seed-vc-realtime)...")
        # Start Audio
        p_audio = subprocess.Popen([audio_python, audio_script], cwd=audio_dir, env=env)
        processes.append(p_audio)
        
        print("Starting Video Project (Deep-Live-Cam)...")
        # Start Video
        p_video = subprocess.Popen([video_python, video_script, "--execution-provider", "cuda"], cwd=video_dir, env=env)
        processes.append(p_video)
        
        print("\n=== Synchronization Active ===")
        print("Press Ctrl+C to stop all processes.\n")
        
        while True:
            # Check if processes are alive
            if p_audio.poll() is not None:
                print("Audio process exited.")
                # Reset audio delay in SHM so video doesn't wait for a ghost
                shm.buf[0:8] = struct.pack('d', 0.0)
                break
            if p_video.poll() is not None:
                print("Video process exited.")
                # Reset video delay
                shm.buf[8:16] = struct.pack('d', 0.0)
                break
                
            # Read Delays (double precision float)
            audio_delay = struct.unpack('d', shm.buf[0:8])[0]
            video_delay = struct.unpack('d', shm.buf[8:16])[0]
            
            # Determine target (Sync to the slowest)
            # We no longer enforce a central target. 
            # Audio syncs to Video, Video syncs to Audio.
            # This prevents "self-sync" jitter and handles single-mode gracefully.
            # target = max(audio_delay, video_delay)
            
            # Enforce a minimum target if both are 0 (startup) or very low
            # if target < 10: 
            #    target = 0
            
            # Write Target (Legacy/Unused)
            # shm.buf[16:24] = struct.pack('d', target)
            
            # print(f"Audio: {audio_delay:6.2f}ms | Video: {video_delay:6.2f}ms", end='\r')
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for p in processes:
            if p.poll() is None:
                print(f"Terminating process {p.pid}...")
                p.terminate()
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    p.kill()
        
        shm.close()
        shm.unlink()
        print("Shared Memory Cleaned up.")

if __name__ == "__main__":
    main()

